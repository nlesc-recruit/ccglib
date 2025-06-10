#ifndef COMPLEX_ORDER_H_
#define COMPLEX_ORDER_H_

namespace ccglib {

/*
 * Location of the complex axis for the input and output data for the GEMM
 * kernel
 * - complex_planar: batch, complex, MNK
 * - complex_interleaved: batch, MNK, complex
 */
enum ComplexAxisLocation { complex_planar, complex_interleaved };

} // namespace ccglib

#endif // COMPLEX_ORDER_H_