"""Thin Windows helpers: window lookup, window capture, screen capture, mouse clicks."""

import ctypes
import threading
import time
from ctypes import wintypes

import mss
import numpy as np

user32 = ctypes.windll.user32
gdi32 = ctypes.windll.gdi32

try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except OSError:
    user32.SetProcessDPIAware()

PW_RENDERFULLCONTENT = 2
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004

user32.GetDC.restype = wintypes.HDC
user32.GetDC.argtypes = [wintypes.HWND]
user32.ReleaseDC.argtypes = [wintypes.HWND, wintypes.HDC]
user32.PrintWindow.argtypes = [wintypes.HWND, wintypes.HDC, wintypes.UINT]
user32.SetWindowPos.argtypes = [wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, wintypes.UINT]
gdi32.CreateCompatibleDC.restype = wintypes.HDC
gdi32.CreateCompatibleDC.argtypes = [wintypes.HDC]
gdi32.CreateCompatibleBitmap.restype = wintypes.HBITMAP
gdi32.CreateCompatibleBitmap.argtypes = [wintypes.HDC, ctypes.c_int, ctypes.c_int]
gdi32.SelectObject.restype = wintypes.HGDIOBJ
gdi32.SelectObject.argtypes = [wintypes.HDC, wintypes.HGDIOBJ]
gdi32.DeleteObject.argtypes = [wintypes.HGDIOBJ]
gdi32.DeleteDC.argtypes = [wintypes.HDC]
gdi32.GetDIBits.argtypes = [
    wintypes.HDC, wintypes.HBITMAP, wintypes.UINT, wintypes.UINT,
    ctypes.c_void_p, ctypes.c_void_p, wintypes.UINT,
]


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", wintypes.DWORD), ("biWidth", wintypes.LONG), ("biHeight", wintypes.LONG),
        ("biPlanes", wintypes.WORD), ("biBitCount", wintypes.WORD), ("biCompression", wintypes.DWORD),
        ("biSizeImage", wintypes.DWORD), ("biXPelsPerMeter", wintypes.LONG),
        ("biYPelsPerMeter", wintypes.LONG), ("biClrUsed", wintypes.DWORD), ("biClrImportant", wintypes.DWORD),
    ]


def find_windows(title):
    """Visible top-level windows whose title equals `title`, as (hwnd, (left, top, right, bottom))."""
    found = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def visit(hwnd, _):
        if user32.IsWindowVisible(hwnd):
            buffer = ctypes.create_unicode_buffer(256)
            user32.GetWindowTextW(hwnd, buffer, 256)
            if buffer.value == title:
                rect = wintypes.RECT()
                user32.GetWindowRect(hwnd, ctypes.byref(rect))
                found.append((hwnd, (rect.left, rect.top, rect.right, rect.bottom)))
        return True

    user32.EnumWindows(visit, 0)
    return found


def set_topmost(hwnd, topmost):
    """Pin a window above all others (or release it) without moving or focusing it."""
    HWND_TOPMOST, HWND_NOTOPMOST = -1, -2
    SWP_NOSIZE, SWP_NOMOVE, SWP_NOACTIVATE = 0x0001, 0x0002, 0x0010
    user32.SetWindowPos(
        wintypes.HWND(hwnd), wintypes.HWND(HWND_TOPMOST if topmost else HWND_NOTOPMOST),
        0, 0, 0, 0, SWP_NOSIZE | SWP_NOMOVE | SWP_NOACTIVATE,
    )


def capture_window(hwnd):
    """BGR image of a window's contents, even when it is covered by other windows."""
    rect = wintypes.RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(rect))
    width, height = rect.right - rect.left, rect.bottom - rect.top
    window_dc = user32.GetDC(hwnd)
    memory_dc = gdi32.CreateCompatibleDC(window_dc)
    bitmap = gdi32.CreateCompatibleBitmap(window_dc, width, height)
    previous = gdi32.SelectObject(memory_dc, bitmap)
    try:
        user32.PrintWindow(hwnd, memory_dc, PW_RENDERFULLCONTENT)
        header = BITMAPINFOHEADER()
        header.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        header.biWidth, header.biHeight = width, -height
        header.biPlanes, header.biBitCount = 1, 32
        pixels = np.empty((height, width, 4), dtype=np.uint8)
        gdi32.GetDIBits(memory_dc, bitmap, 0, height, pixels.ctypes.data, ctypes.byref(header), 0)
    finally:
        gdi32.SelectObject(memory_dc, previous)
        gdi32.DeleteObject(bitmap)
        gdi32.DeleteDC(memory_dc)
        user32.ReleaseDC(hwnd, window_dc)
    return pixels[:, :, :3].copy()


# Screen capture goes through Windows' Desktop Duplication (DXGI), the way recording tools capture:
# a whole 1440p frame takes a few milliseconds there, against close to a hundred through GDI while a
# game is running. One duplicator runs at CAPTURE_FPS for the whole program and every reader cuts
# what it needs from its latest frame. GDI (mss) is the fallback.
CAPTURE_FPS = 60
_screen_lock = threading.Lock()
_duplicator = None
_duplicator_failed = False


def _primary_duplicator():
    global _duplicator, _duplicator_failed
    with _screen_lock:
        if _duplicator is None and not _duplicator_failed:
            try:
                import dxcam

                device = output = 0
                for line in dxcam.output_info().splitlines():
                    if "Primary:True" in line:
                        device = int(line.split("Device[")[1].split("]")[0])
                        output = int(line.split("Output[")[1].split("]")[0])
                camera = dxcam.create(device_idx=device, output_idx=output, output_color="BGR")
                camera.start(target_fps=CAPTURE_FPS, video_mode=True)
                threading.Thread(target=_keep_latest, args=(camera,), daemon=True).start()
                deadline = time.time() + 2
                while _latest_frame is None and time.time() < deadline:
                    time.sleep(0.02)
                _duplicator = camera
            except Exception:  # noqa: BLE001 - any failure here just means falling back to GDI
                _duplicator_failed = True
        return _duplicator


_latest_frame = None


def _keep_latest(camera):
    """Hold on to the newest frame, so readers never wait for the next one."""
    global _latest_frame
    while True:
        frame = camera.get_latest_frame()      # waits for a new frame
        if frame is not None:
            _latest_frame = frame


def capture_screen(region=None, copy=True):
    """BGR image of the primary monitor, or of `region` = (left, top, width, height).

    With copy=False the picture may be a view of the capture buffer, cheaper for
    a reader that only shrinks or converts it straight away.
    """
    camera = _primary_duplicator()
    if camera is not None:
        frame = _latest_frame
        if frame is not None:
            if region is None:
                return frame.copy() if copy else frame
            left, top, width, height = (int(v) for v in region)
            part = frame[top:top + height, left:left + width]
            return part.copy() if copy else part
    with mss.mss() as grabber:
        if region is None:
            area = grabber.monitors[1]
        else:
            area = {"left": region[0], "top": region[1], "width": region[2], "height": region[3]}
        return np.asarray(grabber.grab(area))[:, :, :3].copy()


def screen_size():
    """(width, height) of the primary monitor in pixels."""
    return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)


def click(x, y, settle=0.05):
    user32.SetCursorPos(int(x), int(y))
    time.sleep(settle)
    user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
    user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
