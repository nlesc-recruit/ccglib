#ifndef MMA_COMPLEX_ORDER_H_
#define MMA_COMPLEX_ORDER_H_

namespace ccglib::mma {

/*
 * Location of the complex axis for the input and output data for the GEMM
 * kernel
 * - complex_middle: batch, complex, MNK
 * - complex_last: batch, MNK, complex
 */
enum ComplexAxisLocation { complex_middle, complex_last };

} // namespace ccglib::mma

#endif // MMA_COMPLEX_ORDER_