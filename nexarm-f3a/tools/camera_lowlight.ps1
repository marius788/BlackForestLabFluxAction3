# Read or switch the "Low Light Compensation" checkbox of the USB cameras (DirectShow camera control
# property 19 = auto exposure priority). With it ON a camera may lower its frame rate in dim light.
# The value lives in the camera driver; it can reset when the camera is unplugged or the PC restarts.
#
# Usage:  powershell -ExecutionPolicy Bypass -File camera_lowlight.ps1            (show)
#         powershell -ExecutionPolicy Bypass -File camera_lowlight.ps1 -Set 0     (off for all USB cameras)
#         powershell -ExecutionPolicy Bypass -File camera_lowlight.ps1 -Set 1     (on)
#         ... -NameFilter "icspring"   only cameras whose name contains this text (default)
param(
    [ValidateSet(-1, 0, 1)][int]$Set = -1,
    [string]$NameFilter = "icspring"
)

$ErrorActionPreference = "Stop"

Add-Type -TypeDefinition @'
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Runtime.InteropServices.ComTypes;

namespace NexArmCam
{
    [ComImport, Guid("29840822-5B84-11D0-BD3B-00A0C911CE86"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    interface ICreateDevEnum
    {
        [PreserveSig] int CreateClassEnumerator([In] ref Guid type, [Out] out IEnumMoniker enumMoniker, [In] int flags);
    }

    [ComImport, Guid("55272A00-42CB-11CE-8135-00AA004BB851"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    interface IPropertyBag
    {
        [PreserveSig] int Read([In, MarshalAs(UnmanagedType.LPWStr)] string name, [Out, MarshalAs(UnmanagedType.Struct)] out object value, [In] IntPtr errorLog);
        [PreserveSig] int Write([In, MarshalAs(UnmanagedType.LPWStr)] string name, [In, MarshalAs(UnmanagedType.Struct)] ref object value);
    }

    [ComImport, Guid("C6E13370-30AC-11d0-A18C-00A0C9118956"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    interface IAMCameraControl
    {
        [PreserveSig] int GetRange([In] int property, [Out] out int min, [Out] out int max, [Out] out int step, [Out] out int def, [Out] out int caps);
        [PreserveSig] int Set([In] int property, [In] int value, [In] int flags);
        [PreserveSig] int Get([In] int property, [Out] out int value, [Out] out int flags);
    }

    public static class LowLight
    {
        const int AutoExposurePriority = 19;
        const int FlagsManual = 2;

        // Returns one line per matching camera. setValue < 0 = only read.
        public static List<string> Run(string nameFilter, int setValue)
        {
            var lines = new List<string>();
            Guid systemDeviceEnum = new Guid("62BE5D10-60EB-11d0-BD3B-00A0C911CE86");
            Guid videoInputCategory = new Guid("860BB310-5D01-11d0-BD3B-00A0C911CE86");
            Guid propertyBagId = new Guid("55272A00-42CB-11CE-8135-00AA004BB851");
            Guid baseFilterId = new Guid("56A86895-0AD4-11CE-B03A-0020AF0BA770");

            var devEnum = (ICreateDevEnum)Activator.CreateInstance(Type.GetTypeFromCLSID(systemDeviceEnum));
            IEnumMoniker enumMoniker;
            if (devEnum.CreateClassEnumerator(ref videoInputCategory, out enumMoniker, 0) != 0 || enumMoniker == null)
            {
                lines.Add("No video input devices found.");
                return lines;
            }
            var monikers = new IMoniker[1];
            int index = 0;
            while (enumMoniker.Next(1, monikers, IntPtr.Zero) == 0)
            {
                IMoniker moniker = monikers[0];
                object bagObject;
                moniker.BindToStorage(null, null, ref propertyBagId, out bagObject);
                object nameValue;
                ((IPropertyBag)bagObject).Read("FriendlyName", out nameValue, IntPtr.Zero);
                string name = Convert.ToString(nameValue);
                string prefix = "camera " + index + " (" + name + "): ";
                index++;
                if (name.IndexOf(nameFilter, StringComparison.OrdinalIgnoreCase) < 0)
                {
                    lines.Add(prefix + "skipped (name filter)");
                    continue;
                }
                object filterObject;
                moniker.BindToObject(null, null, ref baseFilterId, out filterObject);
                var control = filterObject as IAMCameraControl;
                if (control == null)
                {
                    lines.Add(prefix + "no camera control interface");
                    continue;
                }
                int value, flags;
                int hr = control.Get(AutoExposurePriority, out value, out flags);
                if (hr != 0)
                {
                    lines.Add(prefix + "low light compensation not supported (0x" + hr.ToString("X8") + ")");
                    continue;
                }
                string text = prefix + "low light compensation = " + value;
                if (setValue >= 0)
                {
                    hr = control.Set(AutoExposurePriority, setValue, FlagsManual);
                    int after, afterFlags;
                    control.Get(AutoExposurePriority, out after, out afterFlags);
                    text += " -> set " + setValue + (hr == 0 ? "" : " FAILED 0x" + hr.ToString("X8")) + ", now " + after;
                }
                lines.Add(text);
                Marshal.ReleaseComObject(filterObject);
            }
            return lines;
        }
    }
}
'@

[NexArmCam.LowLight]::Run($NameFilter, $Set) | ForEach-Object { $_ }
