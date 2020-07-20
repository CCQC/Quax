Here we use Obara-Saika, Hamilton Schaefer, and Head Gordon Pople relations together.
https://pdfs.semanticscholar.org/2875/6bb67b78913f7c4c2e9e4f450b89b638d5d2.pdf
Algorithm:
1. Create auxilliary integals [ss|ss](m) for m = 0 up to La + Lb + Lc + Ld

2. Use VRR to form [xs|ss], where x is in the range 0 up to La + Lb + Lc + Ld

3. Transfer angular momentum to third center to form all [xs|ys] 
where x is in [La, La + Lb] and y is in [Lc, Lc + Ld]

4. Form contractions

5. Perform HGP HRR to make [ab|ys] for all y in [Lc, Lc + Ld]

6. Perform HGP HRR to make [ab|cd]
