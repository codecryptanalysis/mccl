#include <mccl/config/config.hpp>
#include <mccl/core/matrix_ops.hpp>

#include <nmmintrin.h>
#include <cassert>

MCCL_BEGIN_NAMESPACE

namespace detail {

#define WORD_MASK_ONE    (~uint64_t(0))
#define WORD_MASK_BIT(s) ((uint64_t(1)<<s)-1)

#define BLOCK64_MASK0       { WORD_MASK_ONE }
#define BLOCK64_MASK1(s)    { WORD_MASK_BIT(s) }
const uint64_block_t<64> _lastblockmask64[64] = 
{
	BLOCK64_MASK0,     BLOCK64_MASK1( 1), BLOCK64_MASK1( 2), BLOCK64_MASK1( 3), BLOCK64_MASK1( 4), BLOCK64_MASK1( 5), BLOCK64_MASK1( 6), BLOCK64_MASK1( 7),
	BLOCK64_MASK1( 8), BLOCK64_MASK1( 9), BLOCK64_MASK1(10), BLOCK64_MASK1(11), BLOCK64_MASK1(12), BLOCK64_MASK1(13), BLOCK64_MASK1(14), BLOCK64_MASK1(15),
	BLOCK64_MASK1(16), BLOCK64_MASK1(17), BLOCK64_MASK1(18), BLOCK64_MASK1(19), BLOCK64_MASK1(20), BLOCK64_MASK1(21), BLOCK64_MASK1(22), BLOCK64_MASK1(23),
	BLOCK64_MASK1(24), BLOCK64_MASK1(25), BLOCK64_MASK1(26), BLOCK64_MASK1(27), BLOCK64_MASK1(28), BLOCK64_MASK1(29), BLOCK64_MASK1(30), BLOCK64_MASK1(31),
	BLOCK64_MASK1(32), BLOCK64_MASK1(33), BLOCK64_MASK1(34), BLOCK64_MASK1(35), BLOCK64_MASK1(36), BLOCK64_MASK1(37), BLOCK64_MASK1(38), BLOCK64_MASK1(39),
	BLOCK64_MASK1(40), BLOCK64_MASK1(41), BLOCK64_MASK1(42), BLOCK64_MASK1(43), BLOCK64_MASK1(44), BLOCK64_MASK1(45), BLOCK64_MASK1(46), BLOCK64_MASK1(47),
	BLOCK64_MASK1(48), BLOCK64_MASK1(49), BLOCK64_MASK1(50), BLOCK64_MASK1(51), BLOCK64_MASK1(52), BLOCK64_MASK1(53), BLOCK64_MASK1(54), BLOCK64_MASK1(55),
	BLOCK64_MASK1(56), BLOCK64_MASK1(57), BLOCK64_MASK1(58), BLOCK64_MASK1(59), BLOCK64_MASK1(60), BLOCK64_MASK1(61), BLOCK64_MASK1(62), BLOCK64_MASK1(63),
};

#define BLOCK128_MASK0       { WORD_MASK_ONE, WORD_MASK_ONE }
#define BLOCK128_MASK1(s)    { WORD_MASK_BIT(s), 0 }
#define BLOCK128_MASK2(s)    { WORD_MASK_ONE, WORD_MASK_BIT(s) }
const uint64_block_t<128> _lastblockmask128[128] = 
{
	BLOCK128_MASK0,     BLOCK128_MASK1( 1), BLOCK128_MASK1( 2), BLOCK128_MASK1( 3), BLOCK128_MASK1( 4), BLOCK128_MASK1( 5), BLOCK128_MASK1( 6), BLOCK128_MASK1( 7),
	BLOCK128_MASK1( 8), BLOCK128_MASK1( 9), BLOCK128_MASK1(10), BLOCK128_MASK1(11), BLOCK128_MASK1(12), BLOCK128_MASK1(13), BLOCK128_MASK1(14), BLOCK128_MASK1(15),
	BLOCK128_MASK1(16), BLOCK128_MASK1(17), BLOCK128_MASK1(18), BLOCK128_MASK1(19), BLOCK128_MASK1(20), BLOCK128_MASK1(21), BLOCK128_MASK1(22), BLOCK128_MASK1(23),
	BLOCK128_MASK1(24), BLOCK128_MASK1(25), BLOCK128_MASK1(26), BLOCK128_MASK1(27), BLOCK128_MASK1(28), BLOCK128_MASK1(29), BLOCK128_MASK1(30), BLOCK128_MASK1(31),
	BLOCK128_MASK1(32), BLOCK128_MASK1(33), BLOCK128_MASK1(34), BLOCK128_MASK1(35), BLOCK128_MASK1(36), BLOCK128_MASK1(37), BLOCK128_MASK1(38), BLOCK128_MASK1(39),
	BLOCK128_MASK1(40), BLOCK128_MASK1(41), BLOCK128_MASK1(42), BLOCK128_MASK1(43), BLOCK128_MASK1(44), BLOCK128_MASK1(45), BLOCK128_MASK1(46), BLOCK128_MASK1(47),
	BLOCK128_MASK1(48), BLOCK128_MASK1(49), BLOCK128_MASK1(50), BLOCK128_MASK1(51), BLOCK128_MASK1(52), BLOCK128_MASK1(53), BLOCK128_MASK1(54), BLOCK128_MASK1(55),
	BLOCK128_MASK1(56), BLOCK128_MASK1(57), BLOCK128_MASK1(58), BLOCK128_MASK1(59), BLOCK128_MASK1(60), BLOCK128_MASK1(61), BLOCK128_MASK1(62), BLOCK128_MASK1(63),
	BLOCK128_MASK2( 0), BLOCK128_MASK2( 1), BLOCK128_MASK2( 2), BLOCK128_MASK2( 3), BLOCK128_MASK2( 4), BLOCK128_MASK2( 5), BLOCK128_MASK2( 6), BLOCK128_MASK2( 7),
	BLOCK128_MASK2( 8), BLOCK128_MASK2( 9), BLOCK128_MASK2(10), BLOCK128_MASK2(11), BLOCK128_MASK2(12), BLOCK128_MASK2(13), BLOCK128_MASK2(14), BLOCK128_MASK2(15),
	BLOCK128_MASK2(16), BLOCK128_MASK2(17), BLOCK128_MASK2(18), BLOCK128_MASK2(19), BLOCK128_MASK2(20), BLOCK128_MASK2(21), BLOCK128_MASK2(22), BLOCK128_MASK2(23),
	BLOCK128_MASK2(24), BLOCK128_MASK2(25), BLOCK128_MASK2(26), BLOCK128_MASK2(27), BLOCK128_MASK2(28), BLOCK128_MASK2(29), BLOCK128_MASK2(30), BLOCK128_MASK2(31),
	BLOCK128_MASK2(32), BLOCK128_MASK2(33), BLOCK128_MASK2(34), BLOCK128_MASK2(35), BLOCK128_MASK2(36), BLOCK128_MASK2(37), BLOCK128_MASK2(38), BLOCK128_MASK2(39),
	BLOCK128_MASK2(40), BLOCK128_MASK2(41), BLOCK128_MASK2(42), BLOCK128_MASK2(43), BLOCK128_MASK2(44), BLOCK128_MASK2(45), BLOCK128_MASK2(46), BLOCK128_MASK2(47),
	BLOCK128_MASK2(48), BLOCK128_MASK2(49), BLOCK128_MASK2(50), BLOCK128_MASK2(51), BLOCK128_MASK2(52), BLOCK128_MASK2(53), BLOCK128_MASK2(54), BLOCK128_MASK2(55),
	BLOCK128_MASK2(56), BLOCK128_MASK2(57), BLOCK128_MASK2(58), BLOCK128_MASK2(59), BLOCK128_MASK2(60), BLOCK128_MASK2(61), BLOCK128_MASK2(62), BLOCK128_MASK2(63),
};

#define BLOCK256_MASK0       { WORD_MASK_ONE, WORD_MASK_ONE, WORD_MASK_ONE, WORD_MASK_ONE }
#define BLOCK256_MASK1(s)    { WORD_MASK_BIT(s), 0 , 0, 0 }
#define BLOCK256_MASK2(s)    { WORD_MASK_ONE, WORD_MASK_BIT(s), 0, 0 }
#define BLOCK256_MASK3(s)    { WORD_MASK_ONE, WORD_MASK_ONE, WORD_MASK_BIT(s), 0 }
#define BLOCK256_MASK4(s)    { WORD_MASK_ONE, WORD_MASK_ONE, WORD_MASK_ONE, WORD_MASK_BIT(s) }
const uint64_block_t<256> _lastblockmask256[256] = 
{
	BLOCK256_MASK0,     BLOCK256_MASK1( 1), BLOCK256_MASK1( 2), BLOCK256_MASK1( 3), BLOCK256_MASK1( 4), BLOCK256_MASK1( 5), BLOCK256_MASK1( 6), BLOCK256_MASK1( 7),
	BLOCK256_MASK1( 8), BLOCK256_MASK1( 9), BLOCK256_MASK1(10), BLOCK256_MASK1(11), BLOCK256_MASK1(12), BLOCK256_MASK1(13), BLOCK256_MASK1(14), BLOCK256_MASK1(15),
	BLOCK256_MASK1(16), BLOCK256_MASK1(17), BLOCK256_MASK1(18), BLOCK256_MASK1(19), BLOCK256_MASK1(20), BLOCK256_MASK1(21), BLOCK256_MASK1(22), BLOCK256_MASK1(23),
	BLOCK256_MASK1(24), BLOCK256_MASK1(25), BLOCK256_MASK1(26), BLOCK256_MASK1(27), BLOCK256_MASK1(28), BLOCK256_MASK1(29), BLOCK256_MASK1(30), BLOCK256_MASK1(31),
	BLOCK256_MASK1(32), BLOCK256_MASK1(33), BLOCK256_MASK1(34), BLOCK256_MASK1(35), BLOCK256_MASK1(36), BLOCK256_MASK1(37), BLOCK256_MASK1(38), BLOCK256_MASK1(39),
	BLOCK256_MASK1(40), BLOCK256_MASK1(41), BLOCK256_MASK1(42), BLOCK256_MASK1(43), BLOCK256_MASK1(44), BLOCK256_MASK1(45), BLOCK256_MASK1(46), BLOCK256_MASK1(47),
	BLOCK256_MASK1(48), BLOCK256_MASK1(49), BLOCK256_MASK1(50), BLOCK256_MASK1(51), BLOCK256_MASK1(52), BLOCK256_MASK1(53), BLOCK256_MASK1(54), BLOCK256_MASK1(55),
	BLOCK256_MASK1(56), BLOCK256_MASK1(57), BLOCK256_MASK1(58), BLOCK256_MASK1(59), BLOCK256_MASK1(60), BLOCK256_MASK1(61), BLOCK256_MASK1(62), BLOCK256_MASK1(63),
	BLOCK256_MASK2( 0), BLOCK256_MASK2( 1), BLOCK256_MASK2( 2), BLOCK256_MASK2( 3), BLOCK256_MASK2( 4), BLOCK256_MASK2( 5), BLOCK256_MASK2( 6), BLOCK256_MASK2( 7),
	BLOCK256_MASK2( 8), BLOCK256_MASK2( 9), BLOCK256_MASK2(10), BLOCK256_MASK2(11), BLOCK256_MASK2(12), BLOCK256_MASK2(13), BLOCK256_MASK2(14), BLOCK256_MASK2(15),
	BLOCK256_MASK2(16), BLOCK256_MASK2(17), BLOCK256_MASK2(18), BLOCK256_MASK2(19), BLOCK256_MASK2(20), BLOCK256_MASK2(21), BLOCK256_MASK2(22), BLOCK256_MASK2(23),
	BLOCK256_MASK2(24), BLOCK256_MASK2(25), BLOCK256_MASK2(26), BLOCK256_MASK2(27), BLOCK256_MASK2(28), BLOCK256_MASK2(29), BLOCK256_MASK2(30), BLOCK256_MASK2(31),
	BLOCK256_MASK2(32), BLOCK256_MASK2(33), BLOCK256_MASK2(34), BLOCK256_MASK2(35), BLOCK256_MASK2(36), BLOCK256_MASK2(37), BLOCK256_MASK2(38), BLOCK256_MASK2(39),
	BLOCK256_MASK2(40), BLOCK256_MASK2(41), BLOCK256_MASK2(42), BLOCK256_MASK2(43), BLOCK256_MASK2(44), BLOCK256_MASK2(45), BLOCK256_MASK2(46), BLOCK256_MASK2(47),
	BLOCK256_MASK2(48), BLOCK256_MASK2(49), BLOCK256_MASK2(50), BLOCK256_MASK2(51), BLOCK256_MASK2(52), BLOCK256_MASK2(53), BLOCK256_MASK2(54), BLOCK256_MASK2(55),
	BLOCK256_MASK2(56), BLOCK256_MASK2(57), BLOCK256_MASK2(58), BLOCK256_MASK2(59), BLOCK256_MASK2(60), BLOCK256_MASK2(61), BLOCK256_MASK2(62), BLOCK256_MASK2(63),
	BLOCK256_MASK3( 0), BLOCK256_MASK3( 1), BLOCK256_MASK3( 2), BLOCK256_MASK3( 3), BLOCK256_MASK3( 4), BLOCK256_MASK3( 5), BLOCK256_MASK3( 6), BLOCK256_MASK3( 7),
	BLOCK256_MASK3( 8), BLOCK256_MASK3( 9), BLOCK256_MASK3(10), BLOCK256_MASK3(11), BLOCK256_MASK3(12), BLOCK256_MASK3(13), BLOCK256_MASK3(14), BLOCK256_MASK3(15),
	BLOCK256_MASK3(16), BLOCK256_MASK3(17), BLOCK256_MASK3(18), BLOCK256_MASK3(19), BLOCK256_MASK3(20), BLOCK256_MASK3(21), BLOCK256_MASK3(22), BLOCK256_MASK3(23),
	BLOCK256_MASK3(24), BLOCK256_MASK3(25), BLOCK256_MASK3(26), BLOCK256_MASK3(27), BLOCK256_MASK3(28), BLOCK256_MASK3(29), BLOCK256_MASK3(30), BLOCK256_MASK3(31),
	BLOCK256_MASK3(32), BLOCK256_MASK3(33), BLOCK256_MASK3(34), BLOCK256_MASK3(35), BLOCK256_MASK3(36), BLOCK256_MASK3(37), BLOCK256_MASK3(38), BLOCK256_MASK3(39),
	BLOCK256_MASK3(40), BLOCK256_MASK3(41), BLOCK256_MASK3(42), BLOCK256_MASK3(43), BLOCK256_MASK3(44), BLOCK256_MASK3(45), BLOCK256_MASK3(46), BLOCK256_MASK3(47),
	BLOCK256_MASK3(48), BLOCK256_MASK3(49), BLOCK256_MASK3(50), BLOCK256_MASK3(51), BLOCK256_MASK3(52), BLOCK256_MASK3(53), BLOCK256_MASK3(54), BLOCK256_MASK3(55),
	BLOCK256_MASK3(56), BLOCK256_MASK3(57), BLOCK256_MASK3(58), BLOCK256_MASK3(59), BLOCK256_MASK3(60), BLOCK256_MASK3(61), BLOCK256_MASK3(62), BLOCK256_MASK3(63),
	BLOCK256_MASK4( 0), BLOCK256_MASK4( 1), BLOCK256_MASK4( 2), BLOCK256_MASK4( 3), BLOCK256_MASK4( 4), BLOCK256_MASK4( 5), BLOCK256_MASK4( 6), BLOCK256_MASK4( 7),
	BLOCK256_MASK4( 8), BLOCK256_MASK4( 9), BLOCK256_MASK4(10), BLOCK256_MASK4(11), BLOCK256_MASK4(12), BLOCK256_MASK4(13), BLOCK256_MASK4(14), BLOCK256_MASK4(15),
	BLOCK256_MASK4(16), BLOCK256_MASK4(17), BLOCK256_MASK4(18), BLOCK256_MASK4(19), BLOCK256_MASK4(20), BLOCK256_MASK4(21), BLOCK256_MASK4(22), BLOCK256_MASK4(23),
	BLOCK256_MASK4(24), BLOCK256_MASK4(25), BLOCK256_MASK4(26), BLOCK256_MASK4(27), BLOCK256_MASK4(28), BLOCK256_MASK4(29), BLOCK256_MASK4(30), BLOCK256_MASK4(31),
	BLOCK256_MASK4(32), BLOCK256_MASK4(33), BLOCK256_MASK4(34), BLOCK256_MASK4(35), BLOCK256_MASK4(36), BLOCK256_MASK4(37), BLOCK256_MASK4(38), BLOCK256_MASK4(39),
	BLOCK256_MASK4(40), BLOCK256_MASK4(41), BLOCK256_MASK4(42), BLOCK256_MASK4(43), BLOCK256_MASK4(44), BLOCK256_MASK4(45), BLOCK256_MASK4(46), BLOCK256_MASK4(47),
	BLOCK256_MASK4(48), BLOCK256_MASK4(49), BLOCK256_MASK4(50), BLOCK256_MASK4(51), BLOCK256_MASK4(52), BLOCK256_MASK4(53), BLOCK256_MASK4(54), BLOCK256_MASK4(55),
	BLOCK256_MASK4(56), BLOCK256_MASK4(57), BLOCK256_MASK4(58), BLOCK256_MASK4(59), BLOCK256_MASK4(60), BLOCK256_MASK4(61), BLOCK256_MASK4(62), BLOCK256_MASK4(63),
};

#define BLOCK512_MASK0       { WORD_MASK_ONE, WORD_MASK_ONE, WORD_MASK_ONE, WORD_MASK_ONE, WORD_MASK_ONE, WORD_MASK_ONE, WORD_MASK_ONE, WORD_MASK_ONE }
#define BLOCK512_MASK1(s)    { WORD_MASK_BIT(s), 0, 0, 0,  0, 0, 0, 0 }
#define BLOCK512_MASK2(s)    { WORD_MASK_ONE, WORD_MASK_BIT(s), 0, 0,  0, 0, 0, 0 }
#define BLOCK512_MASK3(s)    { WORD_MASK_ONE, WORD_MASK_ONE, WORD_MASK_BIT(s), 0,  0, 0, 0, 0 }
#define BLOCK512_MASK4(s)    { WORD_MASK_ONE, WORD_MASK_ONE, WORD_MASK_ONE, WORD_MASK_BIT(s),  0, 0, 0, 0 }
#define BLOCK512_MASK5(s)    { WORD_MASK_ONE, WORD_MASK_ONE, WORD_MASK_ONE, WORD_MASK_ONE, WORD_MASK_BIT(s), 0, 0, 0 }
#define BLOCK512_MASK6(s)    { WORD_MASK_ONE, WORD_MASK_ONE, WORD_MASK_ONE, WORD_MASK_ONE, WORD_MASK_ONE, WORD_MASK_BIT(s), 0, 0 }
#define BLOCK512_MASK7(s)    { WORD_MASK_ONE, WORD_MASK_ONE, WORD_MASK_ONE, WORD_MASK_ONE, WORD_MASK_ONE, WORD_MASK_ONE, WORD_MASK_BIT(s), 0 }
#define BLOCK512_MASK8(s)    { WORD_MASK_ONE, WORD_MASK_ONE, WORD_MASK_ONE, WORD_MASK_ONE, WORD_MASK_ONE, WORD_MASK_ONE, WORD_MASK_ONE, WORD_MASK_BIT(s) }
const uint64_block_t<512> _lastblockmask512[512] = 
{
	BLOCK512_MASK0,     BLOCK512_MASK1( 1), BLOCK512_MASK1( 2), BLOCK512_MASK1( 3), BLOCK512_MASK1( 4), BLOCK512_MASK1( 5), BLOCK512_MASK1( 6), BLOCK512_MASK1( 7),
	BLOCK512_MASK1( 8), BLOCK512_MASK1( 9), BLOCK512_MASK1(10), BLOCK512_MASK1(11), BLOCK512_MASK1(12), BLOCK512_MASK1(13), BLOCK512_MASK1(14), BLOCK512_MASK1(15),
	BLOCK512_MASK1(16), BLOCK512_MASK1(17), BLOCK512_MASK1(18), BLOCK512_MASK1(19), BLOCK512_MASK1(20), BLOCK512_MASK1(21), BLOCK512_MASK1(22), BLOCK512_MASK1(23),
	BLOCK512_MASK1(24), BLOCK512_MASK1(25), BLOCK512_MASK1(26), BLOCK512_MASK1(27), BLOCK512_MASK1(28), BLOCK512_MASK1(29), BLOCK512_MASK1(30), BLOCK512_MASK1(31),
	BLOCK512_MASK1(32), BLOCK512_MASK1(33), BLOCK512_MASK1(34), BLOCK512_MASK1(35), BLOCK512_MASK1(36), BLOCK512_MASK1(37), BLOCK512_MASK1(38), BLOCK512_MASK1(39),
	BLOCK512_MASK1(40), BLOCK512_MASK1(41), BLOCK512_MASK1(42), BLOCK512_MASK1(43), BLOCK512_MASK1(44), BLOCK512_MASK1(45), BLOCK512_MASK1(46), BLOCK512_MASK1(47),
	BLOCK512_MASK1(48), BLOCK512_MASK1(49), BLOCK512_MASK1(50), BLOCK512_MASK1(51), BLOCK512_MASK1(52), BLOCK512_MASK1(53), BLOCK512_MASK1(54), BLOCK512_MASK1(55),
	BLOCK512_MASK1(56), BLOCK512_MASK1(57), BLOCK512_MASK1(58), BLOCK512_MASK1(59), BLOCK512_MASK1(60), BLOCK512_MASK1(61), BLOCK512_MASK1(62), BLOCK512_MASK1(63),
	BLOCK512_MASK2( 0), BLOCK512_MASK2( 1), BLOCK512_MASK2( 2), BLOCK512_MASK2( 3), BLOCK512_MASK2( 4), BLOCK512_MASK2( 5), BLOCK512_MASK2( 6), BLOCK512_MASK2( 7),
	BLOCK512_MASK2( 8), BLOCK512_MASK2( 9), BLOCK512_MASK2(10), BLOCK512_MASK2(11), BLOCK512_MASK2(12), BLOCK512_MASK2(13), BLOCK512_MASK2(14), BLOCK512_MASK2(15),
	BLOCK512_MASK2(16), BLOCK512_MASK2(17), BLOCK512_MASK2(18), BLOCK512_MASK2(19), BLOCK512_MASK2(20), BLOCK512_MASK2(21), BLOCK512_MASK2(22), BLOCK512_MASK2(23),
	BLOCK512_MASK2(24), BLOCK512_MASK2(25), BLOCK512_MASK2(26), BLOCK512_MASK2(27), BLOCK512_MASK2(28), BLOCK512_MASK2(29), BLOCK512_MASK2(30), BLOCK512_MASK2(31),
	BLOCK512_MASK2(32), BLOCK512_MASK2(33), BLOCK512_MASK2(34), BLOCK512_MASK2(35), BLOCK512_MASK2(36), BLOCK512_MASK2(37), BLOCK512_MASK2(38), BLOCK512_MASK2(39),
	BLOCK512_MASK2(40), BLOCK512_MASK2(41), BLOCK512_MASK2(42), BLOCK512_MASK2(43), BLOCK512_MASK2(44), BLOCK512_MASK2(45), BLOCK512_MASK2(46), BLOCK512_MASK2(47),
	BLOCK512_MASK2(48), BLOCK512_MASK2(49), BLOCK512_MASK2(50), BLOCK512_MASK2(51), BLOCK512_MASK2(52), BLOCK512_MASK2(53), BLOCK512_MASK2(54), BLOCK512_MASK2(55),
	BLOCK512_MASK2(56), BLOCK512_MASK2(57), BLOCK512_MASK2(58), BLOCK512_MASK2(59), BLOCK512_MASK2(60), BLOCK512_MASK2(61), BLOCK512_MASK2(62), BLOCK512_MASK2(63),
	BLOCK512_MASK3( 0), BLOCK512_MASK3( 1), BLOCK512_MASK3( 2), BLOCK512_MASK3( 3), BLOCK512_MASK3( 4), BLOCK512_MASK3( 5), BLOCK512_MASK3( 6), BLOCK512_MASK3( 7),
	BLOCK512_MASK3( 8), BLOCK512_MASK3( 9), BLOCK512_MASK3(10), BLOCK512_MASK3(11), BLOCK512_MASK3(12), BLOCK512_MASK3(13), BLOCK512_MASK3(14), BLOCK512_MASK3(15),
	BLOCK512_MASK3(16), BLOCK512_MASK3(17), BLOCK512_MASK3(18), BLOCK512_MASK3(19), BLOCK512_MASK3(20), BLOCK512_MASK3(21), BLOCK512_MASK3(22), BLOCK512_MASK3(23),
	BLOCK512_MASK3(24), BLOCK512_MASK3(25), BLOCK512_MASK3(26), BLOCK512_MASK3(27), BLOCK512_MASK3(28), BLOCK512_MASK3(29), BLOCK512_MASK3(30), BLOCK512_MASK3(31),
	BLOCK512_MASK3(32), BLOCK512_MASK3(33), BLOCK512_MASK3(34), BLOCK512_MASK3(35), BLOCK512_MASK3(36), BLOCK512_MASK3(37), BLOCK512_MASK3(38), BLOCK512_MASK3(39),
	BLOCK512_MASK3(40), BLOCK512_MASK3(41), BLOCK512_MASK3(42), BLOCK512_MASK3(43), BLOCK512_MASK3(44), BLOCK512_MASK3(45), BLOCK512_MASK3(46), BLOCK512_MASK3(47),
	BLOCK512_MASK3(48), BLOCK512_MASK3(49), BLOCK512_MASK3(50), BLOCK512_MASK3(51), BLOCK512_MASK3(52), BLOCK512_MASK3(53), BLOCK512_MASK3(54), BLOCK512_MASK3(55),
	BLOCK512_MASK3(56), BLOCK512_MASK3(57), BLOCK512_MASK3(58), BLOCK512_MASK3(59), BLOCK512_MASK3(60), BLOCK512_MASK3(61), BLOCK512_MASK3(62), BLOCK512_MASK3(63),
	BLOCK512_MASK4( 0), BLOCK512_MASK4( 1), BLOCK512_MASK4( 2), BLOCK512_MASK4( 3), BLOCK512_MASK4( 4), BLOCK512_MASK4( 5), BLOCK512_MASK4( 6), BLOCK512_MASK4( 7),
	BLOCK512_MASK4( 8), BLOCK512_MASK4( 9), BLOCK512_MASK4(10), BLOCK512_MASK4(11), BLOCK512_MASK4(12), BLOCK512_MASK4(13), BLOCK512_MASK4(14), BLOCK512_MASK4(15),
	BLOCK512_MASK4(16), BLOCK512_MASK4(17), BLOCK512_MASK4(18), BLOCK512_MASK4(19), BLOCK512_MASK4(20), BLOCK512_MASK4(21), BLOCK512_MASK4(22), BLOCK512_MASK4(23),
	BLOCK512_MASK4(24), BLOCK512_MASK4(25), BLOCK512_MASK4(26), BLOCK512_MASK4(27), BLOCK512_MASK4(28), BLOCK512_MASK4(29), BLOCK512_MASK4(30), BLOCK512_MASK4(31),
	BLOCK512_MASK4(32), BLOCK512_MASK4(33), BLOCK512_MASK4(34), BLOCK512_MASK4(35), BLOCK512_MASK4(36), BLOCK512_MASK4(37), BLOCK512_MASK4(38), BLOCK512_MASK4(39),
	BLOCK512_MASK4(40), BLOCK512_MASK4(41), BLOCK512_MASK4(42), BLOCK512_MASK4(43), BLOCK512_MASK4(44), BLOCK512_MASK4(45), BLOCK512_MASK4(46), BLOCK512_MASK4(47),
	BLOCK512_MASK4(48), BLOCK512_MASK4(49), BLOCK512_MASK4(50), BLOCK512_MASK4(51), BLOCK512_MASK4(52), BLOCK512_MASK4(53), BLOCK512_MASK4(54), BLOCK512_MASK4(55),
	BLOCK512_MASK4(56), BLOCK512_MASK4(57), BLOCK512_MASK4(58), BLOCK512_MASK4(59), BLOCK512_MASK4(60), BLOCK512_MASK4(61), BLOCK512_MASK4(62), BLOCK512_MASK4(63),
	BLOCK512_MASK5( 0), BLOCK512_MASK5( 1), BLOCK512_MASK5( 2), BLOCK512_MASK5( 3), BLOCK512_MASK5( 4), BLOCK512_MASK5( 5), BLOCK512_MASK5( 6), BLOCK512_MASK5( 7),
	BLOCK512_MASK5( 8), BLOCK512_MASK5( 9), BLOCK512_MASK5(10), BLOCK512_MASK5(11), BLOCK512_MASK5(12), BLOCK512_MASK5(13), BLOCK512_MASK5(14), BLOCK512_MASK5(15),
	BLOCK512_MASK5(16), BLOCK512_MASK5(17), BLOCK512_MASK5(18), BLOCK512_MASK5(19), BLOCK512_MASK5(20), BLOCK512_MASK5(21), BLOCK512_MASK5(22), BLOCK512_MASK5(23),
	BLOCK512_MASK5(24), BLOCK512_MASK5(25), BLOCK512_MASK5(26), BLOCK512_MASK5(27), BLOCK512_MASK5(28), BLOCK512_MASK5(29), BLOCK512_MASK5(30), BLOCK512_MASK5(31),
	BLOCK512_MASK5(32), BLOCK512_MASK5(33), BLOCK512_MASK5(34), BLOCK512_MASK5(35), BLOCK512_MASK5(36), BLOCK512_MASK5(37), BLOCK512_MASK5(38), BLOCK512_MASK5(39),
	BLOCK512_MASK5(40), BLOCK512_MASK5(41), BLOCK512_MASK5(42), BLOCK512_MASK5(43), BLOCK512_MASK5(44), BLOCK512_MASK5(45), BLOCK512_MASK5(46), BLOCK512_MASK5(47),
	BLOCK512_MASK5(48), BLOCK512_MASK5(49), BLOCK512_MASK5(50), BLOCK512_MASK5(51), BLOCK512_MASK5(52), BLOCK512_MASK5(53), BLOCK512_MASK5(54), BLOCK512_MASK5(55),
	BLOCK512_MASK5(56), BLOCK512_MASK5(57), BLOCK512_MASK5(58), BLOCK512_MASK5(59), BLOCK512_MASK5(60), BLOCK512_MASK5(61), BLOCK512_MASK5(62), BLOCK512_MASK5(63),
	BLOCK512_MASK6( 0), BLOCK512_MASK6( 1), BLOCK512_MASK6( 2), BLOCK512_MASK6( 3), BLOCK512_MASK6( 4), BLOCK512_MASK6( 5), BLOCK512_MASK6( 6), BLOCK512_MASK6( 7),
	BLOCK512_MASK6( 8), BLOCK512_MASK6( 9), BLOCK512_MASK6(10), BLOCK512_MASK6(11), BLOCK512_MASK6(12), BLOCK512_MASK6(13), BLOCK512_MASK6(14), BLOCK512_MASK6(15),
	BLOCK512_MASK6(16), BLOCK512_MASK6(17), BLOCK512_MASK6(18), BLOCK512_MASK6(19), BLOCK512_MASK6(20), BLOCK512_MASK6(21), BLOCK512_MASK6(22), BLOCK512_MASK6(23),
	BLOCK512_MASK6(24), BLOCK512_MASK6(25), BLOCK512_MASK6(26), BLOCK512_MASK6(27), BLOCK512_MASK6(28), BLOCK512_MASK6(29), BLOCK512_MASK6(30), BLOCK512_MASK6(31),
	BLOCK512_MASK6(32), BLOCK512_MASK6(33), BLOCK512_MASK6(34), BLOCK512_MASK6(35), BLOCK512_MASK6(36), BLOCK512_MASK6(37), BLOCK512_MASK6(38), BLOCK512_MASK6(39),
	BLOCK512_MASK6(40), BLOCK512_MASK6(41), BLOCK512_MASK6(42), BLOCK512_MASK6(43), BLOCK512_MASK6(44), BLOCK512_MASK6(45), BLOCK512_MASK6(46), BLOCK512_MASK6(47),
	BLOCK512_MASK6(48), BLOCK512_MASK6(49), BLOCK512_MASK6(50), BLOCK512_MASK6(51), BLOCK512_MASK6(52), BLOCK512_MASK6(53), BLOCK512_MASK6(54), BLOCK512_MASK6(55),
	BLOCK512_MASK6(56), BLOCK512_MASK6(57), BLOCK512_MASK6(58), BLOCK512_MASK6(59), BLOCK512_MASK6(60), BLOCK512_MASK6(61), BLOCK512_MASK6(62), BLOCK512_MASK6(63),
	BLOCK512_MASK7( 0), BLOCK512_MASK7( 1), BLOCK512_MASK7( 2), BLOCK512_MASK7( 3), BLOCK512_MASK7( 4), BLOCK512_MASK7( 5), BLOCK512_MASK7( 6), BLOCK512_MASK7( 7),
	BLOCK512_MASK7( 8), BLOCK512_MASK7( 9), BLOCK512_MASK7(10), BLOCK512_MASK7(11), BLOCK512_MASK7(12), BLOCK512_MASK7(13), BLOCK512_MASK7(14), BLOCK512_MASK7(15),
	BLOCK512_MASK7(16), BLOCK512_MASK7(17), BLOCK512_MASK7(18), BLOCK512_MASK7(19), BLOCK512_MASK7(20), BLOCK512_MASK7(21), BLOCK512_MASK7(22), BLOCK512_MASK7(23),
	BLOCK512_MASK7(24), BLOCK512_MASK7(25), BLOCK512_MASK7(26), BLOCK512_MASK7(27), BLOCK512_MASK7(28), BLOCK512_MASK7(29), BLOCK512_MASK7(30), BLOCK512_MASK7(31),
	BLOCK512_MASK7(32), BLOCK512_MASK7(33), BLOCK512_MASK7(34), BLOCK512_MASK7(35), BLOCK512_MASK7(36), BLOCK512_MASK7(37), BLOCK512_MASK7(38), BLOCK512_MASK7(39),
	BLOCK512_MASK7(40), BLOCK512_MASK7(41), BLOCK512_MASK7(42), BLOCK512_MASK7(43), BLOCK512_MASK7(44), BLOCK512_MASK7(45), BLOCK512_MASK7(46), BLOCK512_MASK7(47),
	BLOCK512_MASK7(48), BLOCK512_MASK7(49), BLOCK512_MASK7(50), BLOCK512_MASK7(51), BLOCK512_MASK7(52), BLOCK512_MASK7(53), BLOCK512_MASK7(54), BLOCK512_MASK7(55),
	BLOCK512_MASK7(56), BLOCK512_MASK7(57), BLOCK512_MASK7(58), BLOCK512_MASK7(59), BLOCK512_MASK7(60), BLOCK512_MASK7(61), BLOCK512_MASK7(62), BLOCK512_MASK7(63),
	BLOCK512_MASK8( 0), BLOCK512_MASK8( 1), BLOCK512_MASK8( 2), BLOCK512_MASK8( 3), BLOCK512_MASK8( 4), BLOCK512_MASK8( 5), BLOCK512_MASK8( 6), BLOCK512_MASK8( 7),
	BLOCK512_MASK8( 8), BLOCK512_MASK8( 9), BLOCK512_MASK8(10), BLOCK512_MASK8(11), BLOCK512_MASK8(12), BLOCK512_MASK8(13), BLOCK512_MASK8(14), BLOCK512_MASK8(15),
	BLOCK512_MASK8(16), BLOCK512_MASK8(17), BLOCK512_MASK8(18), BLOCK512_MASK8(19), BLOCK512_MASK8(20), BLOCK512_MASK8(21), BLOCK512_MASK8(22), BLOCK512_MASK8(23),
	BLOCK512_MASK8(24), BLOCK512_MASK8(25), BLOCK512_MASK8(26), BLOCK512_MASK8(27), BLOCK512_MASK8(28), BLOCK512_MASK8(29), BLOCK512_MASK8(30), BLOCK512_MASK8(31),
	BLOCK512_MASK8(32), BLOCK512_MASK8(33), BLOCK512_MASK8(34), BLOCK512_MASK8(35), BLOCK512_MASK8(36), BLOCK512_MASK8(37), BLOCK512_MASK8(38), BLOCK512_MASK8(39),
	BLOCK512_MASK8(40), BLOCK512_MASK8(41), BLOCK512_MASK8(42), BLOCK512_MASK8(43), BLOCK512_MASK8(44), BLOCK512_MASK8(45), BLOCK512_MASK8(46), BLOCK512_MASK8(47),
	BLOCK512_MASK8(48), BLOCK512_MASK8(49), BLOCK512_MASK8(50), BLOCK512_MASK8(51), BLOCK512_MASK8(52), BLOCK512_MASK8(53), BLOCK512_MASK8(54), BLOCK512_MASK8(55),
	BLOCK512_MASK8(56), BLOCK512_MASK8(57), BLOCK512_MASK8(58), BLOCK512_MASK8(59), BLOCK512_MASK8(60), BLOCK512_MASK8(61), BLOCK512_MASK8(62), BLOCK512_MASK8(63),
};

void m_print(std::ostream& o, const cm_ptr& m, bool transpose)
{
	o << "[";
	if (!transpose)
	{
		for (size_t r = 0; r < m.rows; ++r)
		{
			o << (r==0 ? "[" : " [");
			for (size_t c = 0; c < m.columns; ++c)
				o << m_getbit(m,r,c);
			o << "]" << std::endl;
		}
	}
	else
	{
		for (size_t c = 0; c < m.columns; ++c)
		{
			o << "[";
			for (size_t r = 0; r < m.rows; ++r)
				o << m_getbit(m,r,c);
			o << "]" << std::endl;
		}
	}
	o << "]";
}

void v_print(std::ostream& o, const cv_ptr& v)
{
	o << "[";
	for (size_t c = 0; c < v.columns; ++c)
		o << v_getbit(v,c);
	o << "]";
}

size_t m_hw(const cm_ptr& m)
{
	if (m.columns == 0 || m.rows == 0)
		return 0;
	size_t words = (m.columns + 63)/64;
	uint64_t lwm = lastwordmask(m.columns);
	size_t hw = 0;
	for (size_t r = 0; r < m.rows; ++r)
	{
		auto first1 = m.data(r), last1 = m.data(r) + words - 1;
		for (; first1 != last1; ++first1)
			hw += hammingweight(*first1);
		hw += hammingweight((*first1) & lwm);
	}
	return hw;
}

void m_swapcolumns(const m_ptr& m, size_t c1, size_t c2)
{
	size_t w1 = c1/64, w2 = c2/64, r2 = (c1-c2)%64;
	uint64_t* w1ptr = m.data(0)+w1;
	uint64_t w1mask = uint64_t(1) << (c1%64);
	if (w1 == w2)
	{
		// same word column swap
		for (size_t k = 0; k < m.rows; ++k,w1ptr+=m.stride)
		{
			uint64_t x1 = *w1ptr;
			uint64_t tmp = (x1^rotate_left(x1,r2)) & w1mask;
			*w1ptr = x1 ^ tmp ^ rotate_right(tmp,r2);
		}
	}
	else
	{
		uint64_t* w2ptr = m.data(0)+w2;
		// two word column swap
		for (size_t k = 0; k < m.rows; ++k,w1ptr+=m.stride,w2ptr+=m.stride)
		{
			uint64_t x1 = *w1ptr, x2 = *w2ptr;
			uint64_t tmp = (x1^rotate_left(x2,r2)) & w1mask;
			*w1ptr = x1 ^ tmp;
			*w2ptr = x2 ^ rotate_right(tmp,r2);
		}
	}
}

void m_setcolumns(const m_ptr& m, size_t coloffset, size_t cols, bool b)
{
	if (b)
		m_setcolumns(m,coloffset,cols);
	else
		m_clearcolumns(m,coloffset,cols);
}
void m_setcolumns(const m_ptr& m, size_t coloffset, size_t cols)
{
	if (m.columns == 0 || m.rows == 0 || cols == 0)
		return;
	auto firstword = coloffset/64, firstword2 = firstword+1, lastword = (coloffset+cols-1)/64;
	auto fwm = firstwordmask(coloffset);
	auto lwm = lastwordmask(coloffset+cols);
	if (firstword == lastword)
	{
		fwm = lwm = fwm & lwm;
		firstword2 = firstword;
	}
	for (size_t r = 0; r < m.rows; ++r)
	{
		*(m.data(r)+firstword) |= fwm;
		auto first = m.data(r)+firstword2, last = m.data(r)+lastword;
		for (; first != last; ++first)
			*first |= ~uint64_t(0);
		*first |= lwm;
	}
}
void m_flipcolumns(const m_ptr& m, size_t coloffset, size_t cols)
{
	if (m.columns == 0 || m.rows == 0 || cols == 0)
		return;
	auto firstword = coloffset/64, firstword2 = firstword+1, lastword = (coloffset+cols-1)/64;
	auto fwm = firstwordmask(coloffset);
	auto lwm = lastwordmask(coloffset+cols);
	if (firstword == lastword)
	{
		fwm = fwm & lwm;
		lwm = 0;
		firstword2 = firstword;
	}
	for (size_t r = 0; r < m.rows; ++r)
	{
		*(m.data(r)+firstword) ^= fwm;
		auto first = m.data(r)+firstword2, last = m.data(r)+lastword;
		for (; first != last; ++first)
			*first ^= ~uint64_t(0);
		*first ^= lwm;
	}
}
void m_clearcolumns(const m_ptr& m, size_t coloffset, size_t cols)
{
	if (m.columns == 0 || m.rows == 0 || cols == 0)
		return;
	auto firstword = coloffset/64, firstword2 = firstword+1, lastword = (coloffset+cols-1)/64;
	auto fwm = ~firstwordmask(coloffset);
	auto lwm = ~lastwordmask(coloffset+cols);
	if (firstword == lastword)
	{
		fwm = lwm = fwm | lwm;
		firstword2 = firstword;
	}
	for (size_t r = 0; r < m.rows; ++r)
	{
		*(m.data(r)+firstword) &= fwm;
		auto first = m.data(r)+firstword2, last = m.data(r)+lastword;
		for (; first != last; ++first)
			*first = 0;
		*first &= lwm;
	}
}


/* TRANSPOSE FUNCTIONS */

template<size_t bits = 64>
void block_transpose(uint64_t* dst, size_t dststride, const uint64_t* src, size_t srcstride)
{
	static_assert(0 == (bits&(bits-1)), "bits must be power of 2");
	static_assert(64 >= bits, "bits must not exceed uint64_t bitsize");

	
	// mask of lower half bits
	uint64_t m = (uint64_t(1) << (bits/2))-1;
	unsigned int j = (bits/2);
	uint64_t tmp[bits];

	// first loop iteration, load src store in tmp
//#pragma unroll
	const uint64_t* src2 = src + ((bits/2)*srcstride);
	for (unsigned int k=0;  k<bits/2;  ++k, src+=srcstride, src2+=srcstride)
	{
		// j = (bits/2)
		uint64_t a = *src, b = *src2;
		uint64_t t = ((a>>(bits/2)) ^ b) & m;
		tmp[k] = a ^ (t << (bits/2));
		tmp[k+(bits/2)] = b ^ t;
	}
	j>>=1; m^=m<<j;
	// main loop
	for (;  1 != j;  j>>=1,m^=m<<j)
	{
//#pragma unroll
		for (unsigned int l=0,k=0;  l<bits/2;  ++l)
		{
			uint64_t t = ((tmp[k]>>j) ^ tmp[k+j]) & m;
			tmp[k] ^= t<<j;
			tmp[k+j] ^= t;
			k=(k+j+1)&~j;
		}
	}
	// last loop iteration (j==1), load tmp store in dst
//#pragma unroll
	const uint64_t bitmask = (~uint64_t(0)) >> (64 - bits);
	for (unsigned int k=0;  k<bits;  k += 2)
	{
		uint64_t t = ((tmp[k]>>1) ^ tmp[k+1]) & m;
		uint64_t val = tmp[k] ^ (t<<1);
		*dst ^= (*dst ^ val) & bitmask;
		dst+=dststride;
		val = tmp[k+1] ^ t;
		*dst ^= (*dst ^ val) & bitmask;
		dst+=dststride;
	}
}

template<size_t bits = 64>
inline void block_transpose2(uint64_t* dst, size_t dststride, const uint64_t* src, size_t srcstride)
{
	static_assert(0 == (bits&(bits-1)), "bits must be power of 2");
	static_assert(64 >= bits, "bits must not exceed uint64_t bitsize");

	// mask of lower half bits
	uint64_t m = (uint64_t(1) << (bits/2))-1;
	unsigned int j = (bits/2);
	uint64_t tmp[2*bits];

	// first loop iteration, load src store in tmp
//#pragma unroll
	const uint64_t* src2 = src + ((bits/2)*srcstride);
	for (unsigned int k=0;  k<bits/2;  ++k, src+=srcstride, src2+=srcstride)
	{
		// j = (bits/2)
		uint64_t a1 = *src, b1 = *src2;
		uint64_t t1 = ((a1>>(bits/2)) ^ b1) & m;
		tmp[k] = a1 ^ (t1 << (bits/2));
		tmp[k+(bits/2)] = b1 ^ t1;
		uint64_t a2 = *(src+1), b2 = *(src2+1);
		uint64_t t2 = ((a2>>(bits/2)) ^ b2) & m;
		tmp[k+bits] = a2 ^ (t2 << (bits/2));
		tmp[k+(bits/2)+bits] = b2 ^ t2;
	}
	j>>=1; m^=m<<j;
	// main loop
	for (;  1 != j;  j>>=1,m^=m<<j)
	{
//#pragma unroll
		for (unsigned int l=0,k=0;  l<bits/2;  ++l)
		{
			uint64_t t = ((tmp[k]>>j) ^ tmp[k+j]) & m;
			tmp[k] ^= t<<j;
			tmp[k+j] ^= t;
			uint64_t t2 = ((tmp[k+bits]>>j) ^ tmp[k+j+bits]) & m;
			tmp[k+bits] ^= t2<<j;
			tmp[k+j+bits] ^= t2;
			k=(k+j+1)&~j;
		}
	}
	// last loop iteration (j==1), load tmp store in dst
//#pragma unroll
	const uint64_t bitmask = (~uint64_t(0)) >> (64 - bits);
	for (unsigned int k=0;  k<2*bits;  k += 2)
	{
		uint64_t t = ((tmp[k]>>1) ^ tmp[k+1]) & m;
		uint64_t val = tmp[k] ^ (t<<1);
		*dst ^= (*dst ^ val) & bitmask;
		dst+=dststride;
		val = tmp[k+1] ^ t;
		*dst ^= (*dst ^ val) & bitmask;
		dst+=dststride;
	}
}

template<size_t bits = 64>
inline void block_transpose(uint64_t* dst, size_t dststride, size_t dstrows, const uint64_t* src, size_t srcstride, size_t srcrows)
{
	static_assert(bits >= 4, "bits >= 4");
	static_assert(0 == (bits&(bits-1)), "bits must be power of 2");
	static_assert(sizeof(uint64_t)*8 >= bits, "bits must not exceed uint64_t bitsize");
	assert(dstrows <= bits);
	assert(srcrows <= bits);
	// mask of lower half bits
	uint64_t m = (uint64_t(1) << (bits/2))-1;
	unsigned int j = (bits/2);
	uint64_t tmp[bits+2]; // <= add 2 to avoid incorrect out-of-bounds warning
	// first loop iteration, load src store in tmp
	const uint64_t* src2 = src + ((bits/2)*srcstride);
	for (unsigned int k=0;  k<bits/2;  ++k)
	{
		if (k < srcrows)
		{
			uint64_t a = *src, b = 0;
			src += srcstride;
			if ((k+(bits/2)) < srcrows)
			{
				b = *src2;
				src2 += srcstride;
			}
			uint64_t t = (b ^ (a >> (bits/2))) & m;
			tmp[k] = a ^ (t << (bits/2));
			tmp[k+(bits/2)] = b ^ t;
		}
		else
		{
			tmp[k] = 0;
			tmp[k+(bits/2)] = 0;
		}
	}
	j>>=1; m^=m<<j;
	// main loop
	for (;  1 != j;  j>>=1,m^=m<<j)
	{
		for (unsigned int l=0,k=0;  l<bits/2;  ++l)
		{
			uint64_t t = ((tmp[k]>>j) ^ tmp[k+j]) & m;
			tmp[k] ^= t<<j;
			tmp[k+j] ^= t;
			k=(k+j+1)&~j;
		}
	}
	// last loop iteration (j==1), load tmp store in dst
	const uint64_t bitmask = (~uint64_t(0)) >> (64 - bits);
	unsigned int k=0;
	for (;  k+1 < dstrows;  k += 2)
	{
		uint64_t t = ((tmp[k]>>1) ^ tmp[k+1]) & m;
		uint64_t val = tmp[k] ^ (t<<1);
		*dst ^= (*dst ^ val) & bitmask;
		 dst+=dststride;
		val = tmp[k+1] ^ t;
		*dst ^= (*dst ^ val) & bitmask;
		 dst+=dststride;
	}
	// note both k and bits are even and k < dstrows <= bits
	// so k+1 < bits as well, nevertheless compilers may warn
	if (k < dstrows)
	{
		uint64_t t = ((tmp[k]>>1) ^ tmp[k+1]) & m;
		uint64_t val = tmp[k] ^ (t<<1);
		*dst ^= (*dst ^ val) & bitmask;
	}
}



//template<typename uint64_t>
inline void block_transpose(uint64_t* dst, size_t dststride, size_t dstrows, const uint64_t* src, size_t srcstride, size_t srcrows, size_t bits)
{
	assert(0 == (bits&(bits-1))); // bits must be power of 2
	assert(64 >= bits); // bits must not exceed uint64_t bitsize
	assert(dstrows <= bits);
	assert(srcrows <= bits);
	if (bits < 4)
		bits = 4;
	if (bits > 64) throw std::out_of_range("block_transpose: bits > 64");
	// mask of lower half bits
	uint64_t m = (uint64_t(1) << (bits/2))-1;
	unsigned int j = (bits/2);
	uint64_t tmp[8*sizeof(uint64_t)];
	// first loop iteration, load src store in tmp
	const uint64_t* src2 = src + ((bits/2)*srcstride);
	for (unsigned int k=0;  k<bits/2;  ++k)
	{
		if (k < srcrows)
		{
			uint64_t a = *src, b = 0;
			src += srcstride;
			if ((k+(bits/2)) < srcrows)
			{
				b = *src2;
				src2 += srcstride;
			}
			uint64_t t = (b ^ (a >> (bits/2))) & m;
			tmp[k] = a ^ (t << (bits/2));
			tmp[k+(bits/2)] = b ^ t;
		}
		else
		{
			tmp[k] = 0;
			tmp[k+(bits/2)] = 0;
		}
	}
	j>>=1; m^=m<<j;
	// main loop
	for (;  1 != j;  j>>=1,m^=m<<j)
	{
		for (unsigned l=0,k=0;  l<bits/2;  ++l)
		{
			uint64_t t = ((tmp[k]>>j) ^ tmp[k+j]) & m;
			tmp[k] ^= t<<j;
			tmp[k+j] ^= t;
			k=(k+j+1)&~j;
		}
	}
	// last loop iteration (j==1), load tmp store in dst
	const uint64_t bitmask = (~uint64_t(0)) >> (64 - bits);
	unsigned int k=0;
	for (;  k+1 < dstrows;  k += 2)
	{
		uint64_t t = ((tmp[k]>>1) ^ tmp[k+1]) & m;
		uint64_t val = tmp[k] ^ (t<<1);
		*dst ^= (*dst ^ val) & bitmask; dst+=dststride;
		val = tmp[k+1] ^ t;
		*dst ^= (*dst ^ val) & bitmask; dst+=dststride;
	}
	if (k < dstrows)
	{
		uint64_t t = ((tmp[k]>>1) ^ tmp[k+1]) & m;
		uint64_t val = tmp[k] ^ (t<<1);
		*dst ^= (*dst ^ val) & bitmask;
	}
}



void m_transpose(const m_ptr& dst, const cm_ptr& src)
{
	static const size_t bits = 64;
	if (dst.columns == 0 || dst.rows == 0)
		return;
	if (dst.columns != src.rows || dst.rows != src.columns)
	{
		std::cout << dst.columns << " " << src.rows << "  " << dst.rows << " " << src.columns << std::endl;
		throw std::out_of_range("m_transpose: matrix dimensions do not match");
	}
	if (dst.ptr == src.ptr)
	{
		std::cout << dst.ptr << " " << src.ptr << std::endl;
		throw std::runtime_error("m_transpose: src and dst are equal! cannot transpose inplace");
	}
	// process batch of bits rows
	size_t r = 0;
	for (; r+bits <= src.rows; r += bits)
	{
		// process block of bits columns
		size_t c = 0;
		for (; c+2*bits <= src.columns; c += 2*bits)
		{
			block_transpose2(dst.data(c,r), dst.stride, src.data(r,c), src.stride);
		}
		if (c+bits <= src.columns)
		{
			block_transpose(dst.data(c,r), dst.stride, src.data(r,c), src.stride);
			c += bits;
		}
		// process block of partial C columns
		if (c < src.columns)
		{
			block_transpose(dst.data(c,r), dst.stride, (src.columns % bits), src.data(r,c), src.stride, bits);
		}
	}
	// process last rows
	if (r < src.rows)
	{
		size_t c = 0;
		for (; c+bits <= src.columns; c += bits)
		{
			block_transpose(dst.data(c,r), dst.stride, bits, src.data(r,c), src.stride, (src.rows % bits));
		}
		// process final bits x C submatrix
		if (c < src.columns)
		{
			size_t partialbits = next_pow2<uint32_t>(std::max(src.columns % bits, src.rows % bits));
			if (partialbits == bits)
				block_transpose(dst.data(c,r), dst.stride, (src.columns % bits), src.data(r,c), src.stride, (src.rows % bits));
			else
				block_transpose(dst.data(c,r), dst.stride, (src.columns % bits), src.data(r,c), src.stride, (src.rows % bits), partialbits);
		}
	}
}








template<size_t bits, bool masked>
bool m_isequal(const cm_ptr& m1, const cm_ptr& m2, block_tag<bits,masked>)
{
        if (m1.rows != m2.rows || m1.columns != m2.columns)
                return false;
        if (m1.rows == 0 || m1.columns == 0)
                return true;
                
        const size_t blocks = (m1.columns + bits - 1) / bits - 1;
        const size_t stride1 = m1.stride / (bits/64);
        const size_t stride2 = m2.stride / (bits/64);
        
        auto first1 = make_block_ptr(m1.ptr, block_tag<bits,masked>());
        auto first2 = make_block_ptr(m2.ptr, block_tag<bits,masked>());
        auto lwm = lastwordmask(m1.columns, block_tag<bits,masked>());
        for (size_t r = 0; r < m1.rows; ++r, first1+=stride1, first2+=stride2)
        {
        	auto first1r = first1, last1r = first1r + blocks, first2r = first2;
	        for (; first1r != last1r; ++first1r, ++first2r)
        	        if (*first1r != *first2r)
                	        return false;
        	if ((lwm & *first1r) != (lwm & *first2r))
       	                return false;
	}
        return true;
}
template bool m_isequal(const cm_ptr&, const cm_ptr&, block_tag<64 ,true >);
template bool m_isequal(const cm_ptr&, const cm_ptr&, block_tag<128,true >);
template bool m_isequal(const cm_ptr&, const cm_ptr&, block_tag<256,true >);
template bool m_isequal(const cm_ptr&, const cm_ptr&, block_tag<512,true >);

#define MCCL_MATRIX_BASE_FUNCTION_1OP(func,expr) \
template<size_t bits, bool masked> \
void m_ ## func (const m_ptr& dst, block_tag<bits,masked>) \
{ \
        if (dst.rows == 0 || dst.columns == 0) \
                return; \
        size_t blocks = (dst.columns + bits-1)/bits - (masked?1:0); \
        const size_t stride = dst.stride / (bits/64); \
        auto first1 = make_block_ptr(dst.ptr, block_tag<bits,masked>()); \
        if (!masked)  \
        { \
	        for (size_t r = 0; r < dst.rows; ++r, first1+=stride) \
	        { \
	        	auto first1r = first1, last1r = first1r + blocks; \
		        for (; first1r != last1r; ++first1r) \
		        	*first1r = expr ; \
		} \
	} else { \
                auto lwm = lastwordmask(dst.columns, block_tag<bits,masked>()); \
	        for (size_t r = 0; r < dst.rows; ++r, first1+=stride) \
	        { \
	        	auto first1r = first1, last1r = first1r + blocks; \
		        for (; first1r != last1r; ++first1r) \
		        	*first1r = expr ; \
			auto diff = lwm & (( expr ) ^ *first1r); \
			*first1r ^= diff; \
		} \
	} \
} \
template void m_ ## func (const m_ptr&, block_tag<64 ,true >); \
template void m_ ## func (const m_ptr&, block_tag<64 ,false>); \
template void m_ ## func (const m_ptr&, block_tag<128,true >); \
template void m_ ## func (const m_ptr&, block_tag<128,false>); \
template void m_ ## func (const m_ptr&, block_tag<256,true >); \
template void m_ ## func (const m_ptr&, block_tag<256,false>); \
template void m_ ## func (const m_ptr&, block_tag<512,true >); \
template void m_ ## func (const m_ptr&, block_tag<512,false>);

MCCL_MATRIX_BASE_FUNCTION_1OP(not,~*first1r)
MCCL_MATRIX_BASE_FUNCTION_1OP(clear,*first1r^*first1r)
MCCL_MATRIX_BASE_FUNCTION_1OP(set,*first1r|~*first1r)


#define MCCL_MATRIX_BASE_FUNCTION_2OP(func,expr) \
template<size_t bits, bool masked> \
void m_ ## func (const m_ptr& dst, const cm_ptr& m2, block_tag<bits,masked>) \
{ \
	if (dst.rows != m2.rows || dst.columns != m2.columns) \
		throw std::out_of_range("matrices do not have equal dimensions"); \
        if (dst.rows == 0 || dst.columns == 0) \
                return; \
        size_t blocks = (dst.columns + bits-1)/bits - (masked?1:0); \
        const size_t stride1 = dst.stride / (bits/64); \
        const size_t stride2 =  m2.stride / (bits/64); \
        auto first1 = make_block_ptr(dst.ptr, block_tag<bits,masked>()); \
        auto first2 = make_block_ptr( m2.ptr, block_tag<bits,masked>()); \
        if (!masked)  \
        { \
	        for (size_t r = 0; r < dst.rows; ++r, first1+=stride1, first2+=stride2) \
	        { \
	        	auto first1r = first1, last1r = first1r + blocks; \
	        	auto first2r = first2; \
		        for (; first1r != last1r; ++first1r,++first2r) \
		        	*first1r = expr ; \
		} \
	} else { \
                auto lwm = lastwordmask(dst.columns, block_tag<bits,masked>()); \
	        for (size_t r = 0; r < dst.rows; ++r, first1+=stride1, first2+=stride2) \
	        { \
	        	auto first1r = first1, last1r = first1r + blocks; \
	        	auto first2r = first2; \
		        for (; first1r != last1r; ++first1r,++first2r) \
		        	*first1r = expr ; \
			auto diff = lwm & (( expr ) ^ *first1r); \
			*first1r ^= diff; \
		} \
	} \
} \
template void m_ ## func (const m_ptr&, const cm_ptr&, block_tag<64 ,true >); \
template void m_ ## func (const m_ptr&, const cm_ptr&, block_tag<64 ,false>); \
template void m_ ## func (const m_ptr&, const cm_ptr&, block_tag<128,true >); \
template void m_ ## func (const m_ptr&, const cm_ptr&, block_tag<128,false>); \
template void m_ ## func (const m_ptr&, const cm_ptr&, block_tag<256,true >); \
template void m_ ## func (const m_ptr&, const cm_ptr&, block_tag<256,false>); \
template void m_ ## func (const m_ptr&, const cm_ptr&, block_tag<512,true >); \
template void m_ ## func (const m_ptr&, const cm_ptr&, block_tag<512,false>);

MCCL_MATRIX_BASE_FUNCTION_2OP(copy,*first2r)
MCCL_MATRIX_BASE_FUNCTION_2OP(copynot,~(*first2r))
MCCL_MATRIX_BASE_FUNCTION_2OP(and,(*first1r) & (*first2r))
MCCL_MATRIX_BASE_FUNCTION_2OP(or ,(*first1r) | (*first2r))
MCCL_MATRIX_BASE_FUNCTION_2OP(xor,(*first1r) ^ (*first2r))
MCCL_MATRIX_BASE_FUNCTION_2OP(nand,~((*first1r) & (*first2r)))
MCCL_MATRIX_BASE_FUNCTION_2OP(nor ,~((*first1r) | (*first2r)))
MCCL_MATRIX_BASE_FUNCTION_2OP(nxor,~((*first1r) ^ (*first2r)))
MCCL_MATRIX_BASE_FUNCTION_2OP(andin,(*first1r) & (~*first2r))
MCCL_MATRIX_BASE_FUNCTION_2OP(andni,(~*first1r) & (*first2r))
MCCL_MATRIX_BASE_FUNCTION_2OP(orin ,(*first1r) | (~*first2r))
MCCL_MATRIX_BASE_FUNCTION_2OP(orni ,(~*first1r) | (*first2r))


#define MCCL_MATRIX_BASE_FUNCTION_3OP(func,expr) \
template<size_t bits, bool masked> \
void m_ ## func (const m_ptr& dst, const cm_ptr& m2, const cm_ptr& m3, block_tag<bits,masked>) \
{ \
	if (dst.rows != m2.rows || dst.columns != m2.columns) \
		throw std::out_of_range("matrices do not have equal dimensions"); \
        if (dst.rows == 0 || dst.columns == 0) \
                return; \
        size_t blocks = (dst.columns + bits-1)/bits - (masked?1:0); \
        const size_t stride1 = dst.stride / (bits/64); \
        const size_t stride2 =  m2.stride / (bits/64); \
        const size_t stride3 =  m3.stride / (bits/64); \
        auto first1 = make_block_ptr(dst.ptr, block_tag<bits,masked>()); \
        auto first2 = make_block_ptr( m2.ptr, block_tag<bits,masked>()); \
        auto first3 = make_block_ptr( m3.ptr, block_tag<bits,masked>()); \
        if (!masked)  \
        { \
	        for (size_t r = 0; r < dst.rows; ++r, first1+=stride1, first2+=stride2, first3 += stride3) \
	        { \
	        	auto first1r = first1, last1r = first1r + blocks; \
	        	auto first2r = first2; \
	        	auto first3r = first3; \
		        for (; first1r != last1r; ++first1r,++first2r,++first3r) \
		        	*first1r = expr ; \
		} \
	} else { \
                auto lwm = lastwordmask(dst.columns, block_tag<bits,masked>()); \
	        for (size_t r = 0; r < dst.rows; ++r, first1+=stride1, first2+=stride2, first3 += stride3) \
	        { \
	        	auto first1r = first1, last1r = first1r + blocks; \
	        	auto first2r = first2; \
	        	auto first3r = first3; \
		        for (; first1r != last1r; ++first1r,++first2r,++first3r) \
		        	*first1r = expr ; \
			auto diff = lwm & (( expr ) ^ *first1r); \
			*first1r ^= diff; \
		} \
	} \
} \
template void m_ ## func (const m_ptr&, const cm_ptr&, const cm_ptr&, block_tag<64 ,true >); \
template void m_ ## func (const m_ptr&, const cm_ptr&, const cm_ptr&, block_tag<64 ,false>); \
template void m_ ## func (const m_ptr&, const cm_ptr&, const cm_ptr&, block_tag<128,true >); \
template void m_ ## func (const m_ptr&, const cm_ptr&, const cm_ptr&, block_tag<128,false>); \
template void m_ ## func (const m_ptr&, const cm_ptr&, const cm_ptr&, block_tag<256,true >); \
template void m_ ## func (const m_ptr&, const cm_ptr&, const cm_ptr&, block_tag<256,false>); \
template void m_ ## func (const m_ptr&, const cm_ptr&, const cm_ptr&, block_tag<512,true >); \
template void m_ ## func (const m_ptr&, const cm_ptr&, const cm_ptr&, block_tag<512,false>);

MCCL_MATRIX_BASE_FUNCTION_3OP(and,(*first2r) & (*first3r))
MCCL_MATRIX_BASE_FUNCTION_3OP(or ,(*first2r) | (*first3r))
MCCL_MATRIX_BASE_FUNCTION_3OP(xor,(*first2r) ^ (*first3r))
MCCL_MATRIX_BASE_FUNCTION_3OP(nand,~((*first2r) & (*first3r)))
MCCL_MATRIX_BASE_FUNCTION_3OP(nor ,~((*first2r) | (*first3r)))
MCCL_MATRIX_BASE_FUNCTION_3OP(nxor,~((*first2r) ^ (*first3r)))
MCCL_MATRIX_BASE_FUNCTION_3OP(andin,(*first2r) & (~*first3r))
MCCL_MATRIX_BASE_FUNCTION_3OP(andni,(~*first2r) & (*first3r))
MCCL_MATRIX_BASE_FUNCTION_3OP(orin ,(*first2r) | (~*first3r))
MCCL_MATRIX_BASE_FUNCTION_3OP(orni ,(~*first2r) | (*first3r))

} // namespace detail

MCCL_END_NAMESPACE
