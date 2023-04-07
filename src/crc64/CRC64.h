#ifndef CRC64_H
#define CRC64_H
#include <stdlib.h>
#include <stdint.h>

/*
 * These functions compute the CRC-64 checksum on a block of data
 * and provide a way to combine the checksums on two blocks of data.
 * For more information, see:
 * http://en.wikipedia.org/wiki/Computation_of_CRC
 * http://checksumcrc.blogspot.com/2011/12/should-you-use-crc-or-checksum.html
 * http://crcutil.googlecode.com/files/crc-doc.1.0.pdf
 * http://www.ross.net/crc/download/crc_v3.txt
 * This implementation uses the ECMA-182 polynomial with -1 initialization, and
 * computes the bit-reversed CRC.
 */

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Calculate the CRC64 of the provided buffer using the slow reference
 * implementation (in serial).
 */
uint64_t crc64_slow(const void *input, size_t nbytes);

/*
 * Calculate the CRC64 of the provided buffer (in serial).
 */
uint64_t crc64(const void *input, size_t nbytes);

/*
 * Calculate the CRC64 of the provided buffer, in parallel if possible.
 */
uint64_t crc64_omp(const void *input, size_t nbytes);

/*
 * Calculate the 'check bytes' for the provided CRC64. If these bytes are
 * appended to the original buffer, then the new total CRC64 should be -1.
 */
void crc64_invert(uint64_t cs, void *check_bytes);

/*
 * Given the CRC64 of the first part of a buffer, and the CRC64 and length of
 * the second part of a buffer, calculate the CRC64 of the complete buffer.
 */
uint64_t crc64_combine(uint64_t cs1, uint64_t cs2, size_t nbytes2);

#ifdef __cplusplus
}
#endif

#endif // CRC64_H

