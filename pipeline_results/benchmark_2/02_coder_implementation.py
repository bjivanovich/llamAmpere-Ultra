from ctypes import addressof, cast, POINTER, c_uint64
addr = addressof(memview) + OFF_WRITE_INDEX
ptr = cast(addr, POINTER(c_uint64))
ptr.value