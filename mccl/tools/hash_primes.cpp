#include <mccl/tools/hash_primes.hpp>

MCCL_BEGIN_NAMESPACE

namespace detail {

struct hash_prime_t
{
    uint64_t prime;
    uint64_t muldiv;
    unsigned shift;
};

const hash_prime_t hash_prime_table[] = {
    { 3, 0xaaaaaaaaaaaaaaab, 1 },
    { 5, 0xcccccccccccccccd, 2 },
    { 11, 0x2e8ba2e8ba2e8ba3, 1 },
    { 13, 0x4ec4ec4ec4ec4ec5, 2 },
    { 17, 0xf0f0f0f0f0f0f0f1, 4 },
    { 19, 0xd79435e50d79435f, 4 },
    { 37, 0xdd67c8a60dd67c8b, 5 },
    { 41, 0xc7ce0c7ce0c7ce0d, 5 },
    { 59, 0x8ad8f2fba9386823, 5 },
    { 67, 0xf4898d5f85bb3951, 6 },
    { 73, 0x70381c0e070381c1, 5 },
    { 83, 0x3159721ed7e75347, 4 },
    { 109, 0x964fda6c0964fda7, 6 },
    { 113, 0x90fdbc090fdbc091, 6 },
    { 131, 0x3e88cb3c9484e2b, 1 },
    { 149, 0x1b7d6c3dda338b2b, 4 },
    { 163, 0xc907da4e871146ad, 7 },
    { 179, 0xb70fbb5a19be3659, 7 },
    { 197, 0x14cab88725af6e75, 4 },
    { 227, 0x905a38633e06c43b, 7 },
    { 241, 0x10fef010fef010ff, 4 },
    { 257, 0xff00ff00ff00ff01, 8 },
    { 283, 0x73c9b97112ff186d, 7 },
    { 311, 0x34ae820ed114942b, 6 },
    { 349, 0xbbc8408cd63069a1, 8 },
    { 383, 0x558e5ee9f14b87b, 3 },
    { 421, 0x9baade8e4a2f6e1, 4 },
    { 499, 0x8355ace3c897db1, 4 },
    { 509, 0x10182436517a3753, 5 },
    { 521, 0xfb93e672fa98528d, 9 },
    { 557, 0xeb51599f7ba23d97, 9 },
    { 613, 0xd5d20fde972d8539, 9 },
    { 677, 0xc19b6a41cbd11c5d, 9 },
    { 751, 0xae87ab7648f2b4ab, 9 },
    { 827, 0x9e7dada8b4c75a15, 9 },
    { 941, 0x22d291467611f493, 7 },
    { 1013, 0x8163d282e7fdfa71, 9 },
    { 1031, 0x3f90c2ab542cb1c9, 8 },
    { 1039, 0xfc4ddc06e6210431, 10 },
    { 1151, 0x71e06ac264163dd5, 9 },
    { 1277, 0xcd47f7fb3050301d, 10 },
    { 1409, 0x5d065bef48db7b01, 9 },
    { 1549, 0x2a4eff8113017cc7, 8 },
    { 1709, 0x4cb1f4ea479a23a7, 9 },
    { 1879, 0x22e0cce8b3d7209, 4 },
    { 2029, 0x204cb630b3aab56f, 8 },
    { 2053, 0xff6063c1a6f7a539, 11 },
    { 2069, 0xfd66d2187fb0cfdf, 11 },
    { 2281, 0x3976677a38571775, 9 },
    { 2521, 0x33fdf8144f34e7ef, 9 },
    { 2789, 0x5dfdfb0b1b42ea1, 6 },
    { 3067, 0xaaf1e4c9fed4d8b, 7 },
    { 3373, 0x26dbf2f21c62aa77, 9 },
    { 3727, 0x4656227b39e768e3, 10 },
    { 4091, 0x80280c83e938e1c7, 11 },
    { 4099, 0xffd008fe5050f0d3, 12 },
    { 4513, 0xe8587db3e001d0b1, 12 },
    { 4967, 0x698de3dbec009e55, 11 },
    { 5471, 0x2fea49d68ac91cdf, 10 },
    { 6037, 0xadb10aa4c956f917, 12 },
    { 6659, 0x13aef5a893eeee47, 9 },
    { 7331, 0x8f087c50e00c4abb, 12 },
    { 8081, 0x20708651ec2b35e3, 10 },
    { 8179, 0x80341528987df32b, 12 },
    { 8209, 0x7fbc240cd92ca04b, 12 },
    { 8893, 0x75e90739b7a15971, 12 },
    { 9791, 0xd6311a61bc47d9b9, 13 },
    { 10771, 0x185683878bd30827, 10 },
    { 11887, 0x2c1b22b1d86aa59d, 11 },
    { 13093, 0x5016362905607dc3, 12 },
    { 14411, 0x48c31f3f4b3b3e5f, 12 },
    { 15859, 0x421e61356a2ae7f7, 12 },
    { 16381, 0x4003002401b01441, 12 },
    { 16411, 0x1ff285af99eb10d5, 11 },
    { 17467, 0xf020986cb0c0fe33, 14 },
    { 19219, 0xda3cc43b83b2437b, 14 },
    { 21143, 0xc660be3dc6703dcd, 14 },
    { 23269, 0xb440bbff84137ec1, 14 },
    { 25601, 0xa3d566d373a53e59, 14 },
    { 28163, 0x94edf9828118681, 10 },
    { 30983, 0x875fd67d1cbaa2b1, 14 },
    { 32749, 0x801302d26b3beae5, 14 },
    { 32771, 0xfffa0023ff28051, 11 },
    { 34123, 0x3d75672dc1a04939, 13 },
    { 37537, 0xdf79c89bc472c413, 15 },
    { 41299, 0x32c79c467dd8905b, 13 },
    { 45491, 0xb866c7c97b1cce9f, 15 },
    { 50047, 0x53ceab498d24bb71, 14 },
    { 55051, 0x9860fc3a8981e51d, 15 },
    { 60607, 0x8a68ee54cee3687f, 15 },
    { 65449, 0x4015c766c3ec9567, 14 },
    { 65537, 0xffff0000ffff0001, 16 },
    { 66697, 0x3ee2cd6a686c6c49, 14 },
    { 73369, 0xe4ab43b549fb54d9, 16 },
    { 80737, 0x33f340e0a4e18b69, 14 },
    { 88811, 0xbce8c21906adc6a5, 16 },
    { 97711, 0xabb3d25c2fb1a703, 16 },
    { 107509, 0x9c0dd6ea333d1347, 16 },
    { 118259, 0x8dde4ff3d7c3060b, 16 },
    { 130099, 0x80f511ba3054d93f, 16 },
    { 131059, 0x800340152089537d, 16 },
    { 131101, 0x3ffc60348d060329, 15 },
    { 143111, 0x3a9db86a3a346503, 15 },
    { 157427, 0x6a92475bd63be421, 16 },
    { 173177, 0x30708357121e7601, 15 },
    { 190523, 0xb01e13a2ea7a7b1b, 17 },
    { 209579, 0xa01a9e6cf6fdd093, 17 },
    { 230561, 0x9188aaf708b70ba1, 17 },
    { 253637, 0x844b0a68b9832a6d, 17 },
    { 262133, 0x80016003c80a661d, 17 },
    { 262147, 0xffff40008fff9401, 18 },
    { 279001, 0xf0885f110602cc6f, 18 },
    { 306913, 0x6d542caa4177565b, 17 },
    { 337607, 0xc6c72ed7b6a421e1, 18 },
    { 371383, 0x1696656d5f7b5d5d, 15 },
    { 408539, 0xa443f7f39f78f33f, 18 },
    { 449399, 0x4aaa458ec1ceaee3, 17 },
    { 494369, 0x87bf1af5fe7291ff, 18 },
    { 524269, 0x80013002d206b2d, 14 },
    { 524309, 0xfffd6006e3ede9b, 15 },
    { 543811, 0x7b679e1e15f37ef3, 18 },
    { 598193, 0x702f9bf44af820b5, 18 },
    { 658043, 0xcbf708fedf4830a5, 19 },
    { 723851, 0xb96bf89bc1a56e7f, 19 },
    { 796267, 0xa88f06c4952430e3, 19 },
    { 875893, 0x993c3cb94d66446b, 19 },
    { 963497, 0x1169afabd90e55b, 12 },
    { 1048559, 0x8000880090809989, 19 },
    { 1048583, 0xffff900030ffea91, 20 },
    { 1059847, 0x3f51c372bef0b681, 18 },
    { 1165831, 0x7320509e2cf40373, 19 },
    { 1282417, 0x68a8f3f5cb62720d, 19 },
    { 1410679, 0x17c938492b1d8033, 17 },
    { 1551757, 0x567e793c3d67c8d5, 19 },
    { 1706951, 0x4ea14e3d85495af7, 19 },
    { 1877669, 0x23bd92a21ec515ad, 18 },
    { 2065501, 0x40fb10046a9018ad, 19 },
    { 2097091, 0x20003d007448ddab, 18 },
    { 2097169, 0xffff7800483fd99f, 21 },
    { 2272073, 0xec4a8db5565015c9, 21 },
    { 2499337, 0xd6ce2a2f0e099c3f, 21 },
    { 2749277, 0xc346f1c005a7cbfd, 21 },
    { 3024209, 0x58c31fcc0d9e6e27, 20 },
    { 3326629, 0x142c5909f109e211, 18 },
    { 3659309, 0x92b6b7b6f5977563, 21 },
    { 4025269, 0x85600abb373d0a35, 21 },
    { 4194217, 0x8000ae00ec89418b, 21 },
    { 4194319, 0xffffc4000e0ffcb5, 22 },
    { 4427809, 0xf27fe47e0a1ecbef, 22 },
    { 4870589, 0xdc7446c0edbc0001, 22 },
    { 5357657, 0xc8699e606404a9d1, 22 },
    { 5893423, 0x16c62f0323d86a9d, 19 },
    { 6482783, 0x52d09c22a2a6c2c5, 21 },
    { 7131139, 0x969224b9f2ee14a7, 22 },
    { 7844257, 0x22387b89d0b6e6c9, 20 },
    { 8388593, 0x80000f0001c20035, 22 },
    { 8388617, 0x7ffff70000a1fff5, 22 },
    { 8628709, 0x3e381a0144347401, 21 },
    { 9491579, 0x71202ffc15aa8ed7, 22 },
    { 10440743, 0x66d76d80be20283b, 22 },
    { 11484859, 0xbafbe06bc10df241, 23 },
    { 12633353, 0x153f8727ae48a69f, 20 },
    { 13896689, 0x26a20ce1ae196f1b, 21 },
    { 15286367, 0x463de6229adc3ac1, 22 },
    { 16777199, 0x4000044000484005, 22 },
    { 16777259, 0x7fffea80039c7f65, 23 },
    { 16815031, 0x3fdb2782a6bbdcf, 18 },
    { 18496567, 0x1d0682f07cd39653, 21 },
    { 20346247, 0x698c02c475e2b363, 23 },
    { 22380871, 0xbfe74b3e43622dad, 24 },
    { 24618959, 0x573a965b2a4ae29d, 23 },
    { 27080957, 0x9e98ea30217d46f9, 24 },
    { 29789063, 0x902de8e7602b49bf, 24 },
    { 32768033, 0x831265f0f6332b25, 24 },
    { 33554383, 0x80000c40012c201d, 24 },
    { 33554467, 0x3ffffba0004c8ffb, 23 },
    { 36044849, 0xee4ff9a9bf66a315, 25 },
    { 39649343, 0xd8a5c86f5f11996f, 25 },
    { 43614287, 0x6279e54ed03309b7, 24 },
    { 47975777, 0xb30c1d911abaa2c3, 25 },
    { 52773367, 0xa2c52faa5760812d, 25 },
    { 58050791, 0x49fc82bce4a6e201, 24 },
    { 63855907, 0x868545b3a2bff3a7, 25 },
    { 67108837, 0x800003600016c801, 25 },
    { 67108879, 0xfffffc40000e1, 14 },
    { 70241497, 0xf495391269a38bdd, 26 },
    { 77265649, 0x6f2c8e16aa6631f, 21 },
    { 84992227, 0x65113a5512bbb03b, 25 },
    { 93491471, 0x16f84718a0a70eb7, 23 },
    { 102840697, 0xa70d9f92afdfc6e7, 26 },
    { 113124779, 0x97ddd5cd693b9009, 26 },
    { 124437259, 0x8a0f7c651a40d4a3, 26 },
    { 134217649, 0x4000027800186101, 25 },
    { 134217757, 0x3fffff18000349, 17 },
    { 136880987, 0xfb04e1eb937502d5, 27 },
    { 150569087, 0x72197de6304ec18d, 26 },
    { 165625997, 0x67ba154f3d5602cf, 26 },
    { 182188649, 0xbc982332517906c3, 27 },
    { 200407583, 0xab7304d99725f065, 27 },
    { 220448351, 0x9bdcecafd6e80fd1, 27 },
    { 242493193, 0x8db1911664b43a9b, 27 },
    { 266742517, 0x2033fe0734c100cf, 25 },
    { 268435337, 0x800003b8001ba881, 27 },
    { 268435459, 0xffffffd0000009, 20 },
    { 293416793, 0x3a8d135ea855b9a7, 26 },
    { 322758509, 0xd4e9b93666870913, 28 },
    { 355034363, 0xc18ea843a562de51, 28 },
    { 390537803, 0xaff60d38ccdc017f, 28 },
    { 429591611, 0x9ff6f4123971ced9, 28 },
    { 472550777, 0x48b611cd16c821a9, 27 },
    { 519805879, 0x8433c2dea8227133, 28 },
    { 536870701, 0x8000034c0015bd21, 28 },
    { 536870923, 0x3fffffea0000079, 23 },
    { 571786469, 0xf05e1c6ebf337c69, 29 },
    { 628965121, 0x6d420cdda86fc957, 28 },
    { 691861657, 0xc6a6a2943f78c557, 29 },
    { 761047853, 0xb4977c0e248c0475, 29 },
    { 837152663, 0xa42c9f01685bab0f, 29 },
    { 920867963, 0x254ff580af07aa65, 27 },
    { 1012954807, 0x43d73285d40732c9, 28 },
    { 1073741399, 0x8000035200160c89, 29 },
    { 1073741827, 0xfffffff40000009, 26 },
    { 1114250327, 0x3dac5c552ae6f357, 28 },
    { 1225675387, 0x70221c13db4a31a9, 29 },
    { 1348242989, 0x65f0764d64274f17, 29 },
    { 1483067303, 0x2e560732891567fb, 28 },
    { 1631374093, 0x543f52b3cb63209d, 29 },
    { 1794511519, 0x992d5074cd2f728f, 30 },
    { 1973962681, 0x8b4077a40c3268ff, 30 },
    { 2147482819, 0x8000033d0014f913, 30 },
    { 2147483659, 0x3ffffffa80000079, 29 },
    { 2171358967, 0x7e97b283a02dce23, 30 },
    { 2388494881, 0xe62b15ea9f5d64d1, 31 },
    { 2627344409, 0x689f3867435abcbf, 30 },
    { 2890078907, 0xbe38c3656cde797f, 31 },
    { 3179086811, 0x5676e46dda0a9c29, 30 },
    { 3496995563, 0x9d3542069ea1eabb, 31 },
    { 3846695131, 0x47754c8affb4b0e7, 30 },
    { 4231364689, 0x81ec8b1424028223, 31 },
    { 4294967291, 0x800000028000000d, 31 },
    { 4294967311, 0xfffffff1000000e1, 32 },
    { 4654501183, 0x1d8736deff2f9dc3, 29 },
    { 5119951349, 0xd6c01a9151c69b4d, 32 },
    { 5631946487, 0xc33a46b0d9e5dabd, 32 },
    { 6195141137, 0xb17acbe5f796653b, 32 },
    { 6814655297, 0x2856170cc22268e7, 30 },
    { 7496120963, 0x4956b575302f4e25, 31 },
    { 8245733123, 0x2155f55f7e04e7eb, 30 },
    { 8589931619, 0x40000173a0086de5, 31 },
    { 8589934609, 0xfffffff780000049, 33 },
    { 9070306439, 0xf2712711ebbd3d6f, 33 },
    { 9977337101, 0x6e336ed63f105ac1, 32 },
    { 10975070831, 0x642ec1d6fe99750b, 32 },
    { 12072577973, 0x5b133bd3447bf77d, 32 },
    { 13279835783, 0xa597557d77454f65, 33 },
    { 14607819377, 0x25a264e1a790b7ff, 31 },
    { 16068601361, 0x88da28ff669f8631, 33 },
    { 17179869143, 0x8000000520000035, 33 },
    { 17179869209, 0x1fffffff38000005, 31 },
    { 17675461513, 0x7c693c8936ceb87, 29 },
    { 19443007673, 0xe233b3e08261d35, 30 },
    { 21387308441, 0x19b46bb696d6d08b, 31 },
    { 23526039349, 0xbaf19af8db72f49f, 34 },
    { 25878643327, 0x54f974fa6102082f, 33 },
    { 28466507737, 0x9a7fbd631ad75d03, 34 },
    { 31313158553, 0x8c74208575224115, 34 },
    { 34359738337, 0x100000003e000001, 31 },
    { 34359738421, 0x3ffffffe5800000b, 33 },
    { 34444474423, 0xff5ec6c246b348df, 35 },
    { 37888921883, 0xe8279d68fa1c4a11, 35 },
    { 41677814089, 0x1a6197b477f033ed, 32 },
    { 45845595511, 0xbfdd097cab5bfa15, 35 },
    { 50430155063, 0xae6bda143152247d, 35 },
    { 55473170587, 0x4f484bda3567d77b, 34 },
    { 61020487669, 0x240996d75189ba77, 33 },
    { 67122536521, 0x830b98dda4bd8187, 35 },
    { 68719476719, 0x8000000088000001, 35 },
    { 68719476767, 0x3fffffff84000001, 34 },
    { 73834790179, 0xee43a192ab3597ff, 36 },
    { 81218269213, 0xd89a92e1b4af022d, 36 },
    { 89340096149, 0x6274ce66525e660d, 35 },
    { 98274105773, 0x598175d13e8e735f, 35 },
    { 108101516351, 0x28af358da6f08a87, 34 },
    { 118911667991, 0x93f17cebb93bbb95, 36 },
    { 130802834831, 0x219f9c63f14e8c31, 34 },
    { 137438953427, 0x80000000b4000001, 36 },
    { 137438953481, 0xffffffffb8000001, 37 },
    { 143883118367, 0xf488ce78418b952d, 37 },
    { 158271430211, 0xde4dd2f8c9bafc8d, 37 },
    { 174098573263, 0x650c1a13b193af4b, 36 },
    { 191508430621, 0xb7b8e997a8fab53f, 37 },
    { 210659273707, 0x538298b9241db865, 36 },
    { 231725201177, 0x97d62cf270074b29, 37 },
    { 254897721427, 0x2282217c9dfd2067, 35 },
    { 274877906857, 0x80000000ae000001, 37 },
    { 274877906951, 0xffffffffe4000001, 38 },
    { 280387493587, 0xfaf8395b8051fe49, 38 },
    { 308426242997, 0x3909de7d5d24e32f, 36 },
    { 339268867303, 0xcf69b4b087fb57e9, 38 },
    { 373195754033, 0x5e475221b27fc211, 37 },
    { 410515329481, 0xab6a66c897498323, 38 },
    { 451566862477, 0x9bd5179ece2cb2ff, 38 },
    { 496723548731, 0x8daa7290569f9e79, 38 },
    { 546395903611, 0x80c97f6beaac30af, 38 },
    { 549755813753, 0x8000000087000001, 38 },
    { 549755813911, 0xffffffffd2000001, 39 },
    { 601035493973, 0x7514454ad495f4c5, 38 },
    { 661139043457, 0xd4df099edc141bb9, 39 },
    { 727252947943, 0x60c278bc4268929f, 38 },
    { 799978242751, 0x2bfb4e2703ad8e3f, 37 },
    { 879976067041, 0x9feeeda5190423f5, 39 },
    { 967973673761, 0x122c9b014d3269e9, 36 },
    { 1064771041283, 0x421690bec346d32f, 38 },
    { 1099511627491, 0x800000008e800001, 39 },
    { 1099511627791, 0xfffffffff1000001, 40 },
    { 1171248145411, 0x3c14839625f4372b, 38 },
    { 1288372959959, 0x6d3c9228427e9a4b, 39 },
    { 1417210256053, 0x18d39594c1e9ce15, 37 },
    { 1558931281681, 0xb48e6e685fb6bbfd, 40 },
    { 1714824409849, 0xa424645ee2b0a587, 40 },
    { 1886306850839, 0x95385b3efae4fe8f, 40 },
    { 2074937535989, 0x87a798c4d1867de1, 40 },
    { 2199023255027, 0x8000000083400001, 40 },
    { 2199023255579, 0xfffffffff2800001, 41 },
    { 2282431289597, 0xf6a515c2ed1865c5, 41 },
    { 2510674418557, 0xe038fc82a8e172c9, 41 },
    { 2761741860443, 0x32f5adc09858246b, 39 },
    { 3037916046491, 0xb94ed4eae3311767, 41 },
    { 3341707651199, 0xa87635ecc1c8fbe5, 41 },
    { 3675878416351, 0x26496958b63e105b, 39 },
    { 4043466257993, 0x8b397f140a0b0271, 41 },
    { 4398046510073, 0x8000000080e00001, 41 },
    { 4398046511119, 0xfffffffffc400001, 42 },
    { 4447812883907, 0xfd22b8816a9634d9, 42 },
    { 4892594172313, 0xe61f9075a39fd945, 42 },
    { 5381853589549, 0x6899fbd861414f63, 41 },
    { 5920038948517, 0xbe2f3e4397bf4b6d, 42 },
    { 6512042843531, 0x2b394855297f5ef, 36 },
    { 7163247127919, 0x9d2d641e6510d107, 42 },
    { 7879571840717, 0x8ee3724a2cdb7b05, 42 },
    { 8667529024867, 0x207982b3c31dde01, 40 },
    { 8796093020131, 0x8000000081d00001, 42 },
    { 8796093022237, 0xfffffffffc600001, 43 },
    { 9534281927431, 0xec2de51b5442bb7d, 43 },
    { 10487710120177, 0x6b5aadf5262c3d49, 42 },
    { 11536481132203, 0x30cc2086b41c0d95, 41 },
    { 12690129245477, 0xb171eaa400152265, 43 },
    { 13959142170089, 0xa15049ac59f97bd9, 43 },
    { 15355056387107, 0x92a6146e22e0560b, 43 },
    { 16890562025959, 0x10aa253b0f0035bb, 40 },
    { 17592186040271, 0x8000000081880001, 43 },
    { 17592186044423, 0xffffffffff900001, 44 },
    { 18579618228557, 0x79328321e166d40d, 43 },
    { 20437580051501, 0x3716f5c993fa7f01, 42 },
    { 22481338056769, 0xc8537dc5b880221f, 44 },
    { 24729471862489, 0x5b0ead886a747a1d, 43 },
    { 27202419048751, 0xa58f0cf80707bff5, 44 },
    { 29922660953683, 0x4b4105e519da8ba3, 43 },
    { 32914927049111, 0x111a6a1128a6d581, 41 },
    { 35184372080579, 0x8000000080f40001, 44 },
    { 35184372088891, 0xfffffffffe280001, 45 },
    { 36206419754051, 0xf8c606cb08a0584d, 45 },
    { 39827061729479, 0xe228634435d43eb, 41 },
    { 43809767902501, 0xcd99146c8c8869f1, 45 },
    { 48190744692863, 0xbae8411cdafbe98d, 45 },
    { 53009819162159, 0x54f534de92045259, 44 },
    { 58310801078507, 0x4d3c018483fdba87, 44 },
    { 64141881186403, 0x8c6d1a08356047b5, 45 },
    { 70368744161279, 0x8000000080020001, 45 },
    { 70368744177679, 0xffffffffffc40001, 46 },
    { 70556069305063, 0xff5200c91aec4f19, 46 },
    { 77611676235623, 0x3a07002db47dece7, 44 },
    { 85372843859197, 0x1a6045e63ac0e697, 43 },
    { 93910128245123, 0xbfd370b8f10fac9d, 46 },
    { 103301141069683, 0xae6320a8209e7063, 46 },
    { 113631255176683, 0x27a22a54c18c3731, 44 },
    { 124994380694531, 0x901f25914a8e6bd5, 46 },
    { 137493818764019, 0x83050ae12c5ce861, 46 },
    { 140737488322549, 0x80000000800b0001, 46 },
    { 140737488355333, 0xfffffffffff60001, 47 },
    { 151243200640507, 0xee37b6b0ad2b167d, 47 },
    { 166367520704569, 0x6c47dead65f439e5, 46 },
    { 183004272775039, 0x626fe1b4e84aa3c9, 46 },
    { 201304700052563, 0xb2f9f77777b9e0c5, 47 },
    { 221435170057841, 0xa2b4b26c9b51a165, 47 },
    { 243578687063749, 0x24fa85a451c3b065, 45 },
    { 267936555770167, 0x8677b76ccc242d67, 47 },
    { 281474976645119, 0x8000000080008001, 47 },
    { 281474976710677, 0xffffffffffeb0001, 48 },
    { 294730211347217, 0x1e8f926a2e61843b, 45 },
    { 324203232481963, 0x6f215a3c4b87fe19, 47 },
    { 356623555730203, 0xca0e186da07ef1eb, 48 },
    { 392285911303303, 0x5bd7dc8eebc8184b, 47 },
    { 431514502433683, 0xa6fcd6d54f853521, 48 },
    { 474665952677113, 0x12f9d2983d63b2a9, 45 },
    { 522132547945039, 0x8a019e81be3996f1, 48 },
    { 562949953290239, 0x8000000080004001, 48 },
    { 562949953421381, 0xffffffffffdd8001, 49 },
    { 574345802739623, 0x3ebaeaf527e20335, 47 },
    { 631780383013631, 0xe41c1092bf80cff3, 49 },
    { 694958421315013, 0xcf5f54e27f862c87, 49 },
    { 764454263446547, 0xbc8535e52e1404d3, 49 },
    { 840899689791241, 0xab61d3e79e37f00d, 49 },
    { 924989658770389, 0x9bcd4c46ece89ba9, 49 },
    { 1017488624647447, 0x46d1ae4ec8c55c33, 48 },
    { 1119237487112377, 0x406187305961e16f, 48 },
    { 0, 0, 0 }
};


void hash_prime::_check()
{
    if (_prime == 0)
    {
        _muldiv = 0;
        _shift = 0;
        return;
    }
    // basically _muldiv must satisfy the equation
    //   _muldiv = ((uint128_t(1)<<(64+_shift)) / _prime) + 1
    // but must also pass the following tests to ensure correctness for all uint64_t input values
    uint128_t bigint = uint128_t(_muldiv) * _prime;
    if (uint64_t(bigint>>64) != (uint64_t(1)<<_shift))
        throw std::runtime_error("hash_prime::_check(): invalid parameters (fail 1)");
    if (uint64_t(bigint) >= _prime)
        throw std::runtime_error("hash_prime::_check(): invalid parameters (fail 2)");
    // check validity by checking correct results for 6 specific values
    if (mod(1) != 1)
        throw std::runtime_error("hash_prime::_check(): invalid parameters (fail 3)");
    if (mod(_prime-1) != _prime-1)
        throw std::runtime_error("hash_prime::_check(): invalid parameters (fail 4)");
    if (mod(_prime) != 0)
        throw std::runtime_error("hash_prime::_check(): invalid parameters (fail 5)");
    uint64_t maxint = ~uint64_t(0);
    if (mod(maxint) != (maxint % _prime))
        throw std::runtime_error("hash_prime::_check(): invalid parameters (fail 6)");
    maxint -= (maxint % _prime);
    if (mod(maxint) != (maxint % _prime))
        throw std::runtime_error("hash_prime::_check(): invalid parameters (fail 7)");
    --maxint;
    if (mod(maxint) != (maxint % _prime))
        throw std::runtime_error("hash_prime::_check(): invalid parameters (fail 8)");
}

// create your own hash_prime for given p
// if dothrow == true then throws when it fails, otherwise it returns hash_prime(0,0,0)
hash_prime create_hash_prime(uint64_t p, bool dothrow)
{
    typedef hash_prime::uint128_t uint128_t;
    for (unsigned shift = 0; (uint64_t(1)<<shift) <= p; ++shift)
    {
        uint128_t n(uint64_t(1)<<shift);
        n <<= 64;
        uint64_t muldiv = uint64_t(n / p) + 1;
        // check if muldiv and shift are correct for all input values
        uint128_t check128 = uint128_t(muldiv) * p;
        if (uint64_t(check128) > p)
            continue;
        if ((check128>>64) != (n>>64))
            continue;
        uint64_t check1 = ~uint64_t(0);
        uint64_t check2 = check1 - (check1%p) - 1;
        if ( (check1/p) != uint64_t( (uint128_t(muldiv)*check1)>> 64)>>shift )
            continue;
        if ( (check2/p) != uint64_t( (uint128_t(muldiv)*check2)>> 64)>>shift )
            continue;
        // muldiv and shift are fine, return hash_prime
        return hash_prime(p, muldiv, shift);
    }
    if (dothrow)
        throw std::runtime_error("create_hash_prime(): failed to create hash_prime");
    return hash_prime(0,0,0);
}

// obtain smallest internal hash_prime with prime > n
hash_prime get_hash_prime_gt(uint64_t n)
{
    auto it = detail::hash_prime_table + 0;
    for (; it->prime != 0; ++it)
        if (it->prime > n)
            return hash_prime(it->prime, it->muldiv, it->shift);
    throw std::runtime_error("get_hash_prime_gt(): could not find suitable hash_prime");
}

// obtain smallest internal hash_prime with prime >= n
hash_prime get_hash_prime_ge(uint64_t n)
{
    auto it = detail::hash_prime_table + 0;
    for (; it->prime != 0; ++it)
        if (it->prime >= n)
            return hash_prime(it->prime, it->muldiv, it->shift);
    throw std::runtime_error("get_hash_prime_ge(): could not find suitable hash_prime");
}

// obtain largest internal hash_prime with prime < n
hash_prime get_hash_prime_lt(uint64_t n)
{
    auto begin = detail::hash_prime_table + 0;
    auto it = begin;
    for (; it->prime != 0 && it->prime < n; ++it)
        ;
    if (it == begin)
        throw std::runtime_error("get_hash_prime_lt(): could not find suitable hash_prime");
    --it;
    return hash_prime(it->prime, it->muldiv, it->shift);
}

// obtain largest internal hash_prime with prime <= n
hash_prime get_hash_prime_le(uint64_t n)
{
    auto begin = detail::hash_prime_table + 0;
    auto it = begin;
    for (; it->prime != 0 && it->prime <= n; ++it)
        ;
    if (it == begin)
        throw std::runtime_error("get_hash_prime_lt(): could not find suitable hash_prime");
    --it;
    return hash_prime(it->prime, it->muldiv, it->shift);
}

} // namespace detail

MCCL_END_NAMESPACE
