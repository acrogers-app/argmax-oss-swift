//  um fork: regression tests for float16 MLMultiArray handling.
//  These pass on arm64 and on x86_64 (Intel), where FloatType == Float but Core ML arrays are float16.
//  Before the fix, x86_64 crashed in MLMultiArray(shape:dataType:initialValue:) with EXC_BAD_ACCESS.

import CoreML
import XCTest
@testable import ArgmaxCore

final class X86Float16Tests: XCTestCase {
    func testElementByteSize() {
        XCTAssertEqual(MLMultiArrayDataType.float16.elementByteSize, 2)
        XCTAssertEqual(MLMultiArrayDataType.float32.elementByteSize, 4)
        XCTAssertEqual(MLMultiArrayDataType.int32.elementByteSize, 4)
        XCTAssertEqual(MLMultiArrayDataType.double.elementByteSize, 8)
    }

    func testHalfRoundTrip() {
        for v: Float in [0, 1, -1, 0.5, -10000, 65504, -.infinity] {
            XCTAssertEqual(HalfFloat.value(HalfFloat.bits(v)), v)
        }
    }

    /// Large KV-cache-sized array: an out-of-bounds write here crashed on Intel.
    func testFloat16InitialValueFillsExactlyCount() throws {
        let shape: [NSNumber] = [1, 768, 1, 448]
        let a = try MLMultiArray(shape: shape, dataType: .float16, initialValue: FloatType(-10000))
        XCTAssertEqual(a.dataType, .float16)
        XCTAssertEqual(a[0].floatValue, -10000)
        XCTAssertEqual(a[a.count - 1].floatValue, -10000)
        XCTAssertEqual(a[a.count / 2].floatValue, -10000)
    }

    func testFillLastDimensionFloat16() throws {
        let a = try MLMultiArray(shape: [1, 1, 16], dataType: .float16, initialValue: FloatType(0))
        a.fillLastDimension(indexes: 4..<8, with: -FloatType.infinity)
        for i in 0..<16 {
            let v = a[[0, 0, NSNumber(value: i)]].floatValue
            XCTAssertEqual(v, (4..<8).contains(i) ? -.infinity : 0, "index \(i)")
        }
    }

    func testFloat32IfHalfConvertsValues() throws {
        let a = try MLMultiArray(shape: [1, 1, 8], dataType: .float16, initialValue: FloatType(0))
        for i in 0..<8 { a[i] = NSNumber(value: Float(i) - 3.5) }
        let b = a.float32IfHalf()
        XCTAssertEqual(b.dataType, .float32)
        for i in 0..<8 { XCTAssertEqual(b[i].floatValue, Float(i) - 3.5) }
    }
}
