# References

## ISD algorithms (original papers)

- [[Pra62]](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=1057777) Eugene Prange. The use of information sets in decoding cyclic codes. In: IRE Transactions on Information Theory 8.5 (1962), p. 5-9.
- [[LB88]](https://link.springer.com/content/pdf/10.1007/3-540-45961-8_25.pdf) Pil J. Lee and Ernest F. Brickell. An Observation on the Security of McEliece's Public-Key Cryptosystem. In: Advances in Cryptology - EUROCRYPT'88. T. 330. LNCS. Springer, 1988, p. 275-280.
- [[Ste88]](https://link.springer.com/content/pdf/10.1007%2FBFb0019850.pdf) Jacques Stern. A method for finding codewords of small weight. In: Coding Theory and Applications. Ed: G. D. Cohen and J. Wolfmann. T. 388. LNCS. Springer, 1988, p. 106-113.
- [[Dum91]](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=556688) Ilya Dumer. On minimum distance decoding of linear codes. In: Proc. 5th Joint Soviet-Swedish Int. Workshop Inform. Theory. Moscow, 1991, p. 50_52.
- [[FS09]](https://link.springer.com/content/pdf/10.1007/978-3-642-10366-7_6.pdf) Finiasz, M., & Sendrier, N. (2009, December). Security bounds for the design of code-based cryptosystems. In International Conference on the Theory and Application of Cryptology and Information Security (pp. 88-105). Springer, Berlin, Heidelberg.
- [[MMT11]](https://www.iacr.org/archive/asiacrypt2011/70730106/70730106.pdf) May, A., Meurer, A., & Thomae, E. (2011, December). Decoding Random Linear Codes in $\tilde {\mathcal {O}}(2^{0.054 n}) $. In International Conference on the Theory and Application of Cryptology and Information Security (pp. 107-124). Springer, Berlin, Heidelberg.
- [[BJMM12]](https://link.springer.com/content/pdf/10.1007/978-3-642-29011-4_31.pdf) Anja Becker, Antoine Joux, Alexander May and Alexander Meurer. Decoding Random Binary Linear Codes in 2^{n/20}: How 1 + 1 = 0 Improves Information Set Decoding. In: Advances in Cryptology - EUROCRYPT 2012. LNCS. Springer, 2012.
- [[MO15]](https://link.springer.com/content/pdf/10.1007/978-3-662-46800-5_9.pdf) Alexander May and Ilya Ozerov. "On computing nearest neighbors with applications to decoding of binary linear codes." Annual International Conference on the Theory and Applications of Cryptographic Techniques. Springer, 2015.
- [[BM17]](https://www.cits.ruhr-uni-bochum.de/imperia/md/content/may/paper/bjmm+.pdf) Both, L., & May, A. (2017, September). Optimizing BJMM with nearest neighbors: full decoding in 22/21n and McEliece security. In WCC workshop on coding and cryptography.
- [[BM18]](https://eprint.iacr.org/2017/1139.pdf) Both, L., & May, A. (2018, April). Decoding linear codes with high error rate and its impact for LPN security. In International Conference on Post-Quantum Cryptography (pp. 25-46). Springer, Cham.

## Articles concerning practical aspects of ISD
- [[CC98]](https://ieeexplore.ieee.org/document/651067) Canteaut, A., & Chabaud, F. (1998). A new algorithm for finding minimum-weight words in a linear code: Application to McEliece's cryptosystem and to narrow-sense BCH codes of length 511. IEEE Transactions on Information Theory, 44(1), 367-378.
- [[BLT08]](https://cr.yp.to/codes/mceliece-20080807.pdf) Bernstein, D. J., Lange, T., & Peters, C. (2008, October). Attacking and defending the McEliece cryptosystem. In International Workshop on Post-Quantum Cryptography (pp. 31-46). Springer, Berlin, Heidelberg.
- [[HS13]](https://eprint.iacr.org/2013/162.pdf) Hamdaoui, Y., & Sendrier, N. (2013). A Non Asymptotic Analysis of Information Set Decoding. IACR Cryptol. ePrint Arch., 2013, 162.

## Surveys and PhD thesis
- [PhD thesis of Alexander Meurer](http://www-brs.ub.ruhr-uni-bochum.de/netahtml/HSS/Diss/MeurerAlexander/diss.pdf) "A Coding-Theoretic Approach to Cryptanalysis" supervised by Alexander May.
- [PhD thesis of Gregory Landais](https://tel.archives-ouvertes.fr/tel-01142563/document) (in French) "Mise en oeuvre de cryptosystèmes basés sur les codes correcteurs d’erreurs et de leurs cryptanalyses" supervised by Nicolas Sendrier.
- [PhD thesis of David Hobach](https://hackingthe.net/downloads/isd.pdf) "Practical Analysis of Information Set Decoding Algorithms" supervised by Alexander Meurer.
- [PhD thesis of Anja Becker](https://lqsn.fr/docs/isd/these_becker.pdf) "The representation technique: Applications to hard problems in cryptography" supervised by Antoine Joux.
- [PhD thesis of Christiane Peters](https://christianepeters.files.wordpress.com/2012/10/20110510-diss.pdf) "Curves, Codes, and Cryptography" supervised by Tanja Lange and Daniel J. Bernstein.
- [PhD thesis of Kevin Carrier](https://hal.archives-ouvertes.fr/tel-02955488v3/document) (in French) "Recherche de presque-collisions pour le décodage et la reconnaissance de codes correcteurs" supervised by Jean-Pierre Tillich and Nicolas Sendrier.
- [Slides](https://lqsn.fr/docs/isd/MOOC-ISD.pdf) used by Nicolas Sendrier in his [MOOC on code-based crypto](https://www.fun-mooc.fr/courses/course-v1:inria+41006+archiveouvert/about)

## Challenges
- [Decoding challenge website](http://decodingchallenge.org/) by Aragon, Lavauzelle, Lequesne (2019)
- [Wild McElice](https://pqcrypto.org/wild-challenges.html) by Bernstein, Lange and Peters (2013, not maintained)

## Existing implementations of ISD or other useful tools
- [M4RI library](https://bitbucket.org/malb/m4ri/), by Martin Albrecht and Gregory Bard: fast arithmetic with dense matrices over F_2 (implementing the method of the four Russians)
- [CaWof](https://github.com/cr-marcstevens/inria-cawof), by Rodolfo Canto-Torres: calculator of asymptotic complexity for ISD algorithms Prange, Stern, Dummer, MMT and BJMM
- [ISD implementation](https://github.com/cr-marcstevens/inria-collision-dec) by Gregory Landais (2013): Prange, Dumer, MMT, BJMM
- [ISD implementation](https://gitlab.inria.fr/vvasseur/isd) by Valentin Vasseur (2020): Dumer
