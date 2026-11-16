//  um fork: x86_64 support for float16 MLMultiArrays.
//  On Intel Macs `FloatType == Float` (Swift has no Float16 on macOS x86_64), but Core ML
//  models still exchange `.float16` arrays. Every raw-memory access must therefore use the
//  array's real element size, and half-floats must be converted explicitly.

import Accelerate
import CoreML

public extension MLMultiArrayDataType {
    /// Size in bytes of one element of this data type.
    var elementByteSize: Int {
        switch self {
            case .float16: return 2
            case .float32, .int32: return 4
            case .double: return 8
            default: return MemoryLayout<FloatType>.stride
        }
    }
}

public enum HalfFloat {
    /// IEEE 754 binary16 bit pattern for a Float (round-to-nearest via vImage).
    public static func bits(_ value: Float) -> UInt16 {
        var src = value
        var dst: UInt16 = 0
        withUnsafeMutablePointer(to: &src) { sp in
            withUnsafeMutablePointer(to: &dst) { dp in
                var s = vImage_Buffer(data: sp, height: 1, width: 1, rowBytes: 4)
                var d = vImage_Buffer(data: dp, height: 1, width: 1, rowBytes: 2)
                _ = vImageConvert_PlanarFtoPlanar16F(&s, &d, 0)
            }
        }
        return dst
    }

    /// Float value of an IEEE 754 binary16 bit pattern.
    public static func value(_ bits: UInt16) -> Float {
        var src = bits
        var dst: Float = 0
        withUnsafeMutablePointer(to: &src) { sp in
            withUnsafeMutablePointer(to: &dst) { dp in
                var s = vImage_Buffer(data: sp, height: 1, width: 1, rowBytes: 2)
                var d = vImage_Buffer(data: dp, height: 1, width: 1, rowBytes: 4)
                _ = vImageConvert_Planar16FtoPlanarF(&s, &d, 0)
            }
        }
        return dst
    }
}

public extension MLMultiArray {
    /// Returns a contiguous `.float32` copy when this array is `.float16`, otherwise `self`.
    /// Used on x86_64 so downstream code that reads elements as `FloatType` (= Float) is correct.
    func float32IfHalf() -> MLMultiArray {
        guard dataType == .float16,
              let out = try? MLMultiArray(shape: shape, dataType: .float32) else { return self }
        let n = count
        withUnsafeBytes { src in
            var s = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: src.baseAddress!), height: 1, width: vImagePixelCount(n), rowBytes: n * 2)
            var d = vImage_Buffer(data: out.dataPointer, height: 1, width: vImagePixelCount(n), rowBytes: n * 4)
            _ = vImageConvert_Planar16FtoPlanarF(&s, &d, 0)
        }
        return out
    }
}
