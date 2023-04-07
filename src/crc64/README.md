These functions compute the CRC-64 checksum on a block of data
and provide a way to combine the checksums on two blocks of data.
For more information, see:

http://en.wikipedia.org/wiki/Computation_of_CRC

http://checksumcrc.blogspot.com/2011/12/should-you-use-crc-or-checksum.html

http://crcutil.googlecode.com/files/crc-doc.1.0.pdf

http://www.ross.net/crc/download/crc_v3.txt

This implementation uses the ECMA-182 polynomial with -1 initialization, and computes the bit-reversed CRC.
