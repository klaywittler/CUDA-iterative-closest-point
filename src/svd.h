/**************************************************************************
**
**  svd3
**
** Quick singular value decomposition as described by:
** A. McAdams, A. Selle, R. Tamstorf, J. Teran and E. Sifakis,
** "Computing the Singular Value Decomposition of 3x3 matrices
** with minimal branching and elementary floating point operations",
**  University of Wisconsin - Madison technical report TR1690, May 2011
**
**	OPTIMIZED CPU VERSION
** 	Implementation by: Eric Jang
**
**  13 Apr 2014
**
**************************************************************************/


#ifndef SVD_H
#define SVD_H

#define _gamma 5.828427124 // FOUR_GAMMA_SQUARED = sqrt(8)+3;
#define _cstar 0.923879532 // cos(pi/8)
#define _sstar 0.3826834323 // sin(p/8)
//#define EPSILON 1e-6

#include "math.h"
#include <glm/glm.hpp>
#include <glm/gtx/quaternion.hpp>

/* This is a novel and fast routine for the reciprocal square root of an
IEEE float (single precision).
http://www.lomont.org/Math/Papers/2003/InvSqrt.pdf
http://playstation2-linux.com/download/p2lsd/fastrsqrt.pdf
http://www.beyond3d.com/content/articles/8/
*/
//inline float rsqrt(float x) {
//	// int ihalf = *(int *)&x - 0x00800000; // Alternative to next line,
//	// float xhalf = *(float *)&ihalf;      // for sufficiently large nos.
//	float xhalf = 0.5f*x;
//	int i = *(int *)&x;          // View x as an int.
//	// i = 0x5f3759df - (i >> 1);   // Initial guess (traditional).
//	i = 0x5f375a82 - (i >> 1);   // Initial guess (slightly better).
//	x = *(float *)&i;            // View i as float.
//	x = x*(1.5f - xhalf*x*x);    // Newton step.
//	// x = x*(1.5008908 - xhalf*x*x);  // Newton step for a balanced error.
//	return x;
//}

/* This is rsqrt with an additional step of the Newton iteration, for
increased accuracy. The constant 0x5f37599e makes the relative error
range from 0 to -0.00000463.
You can't balance the error by adjusting the constant. */
inline float rsqrt1(float x) {
	float xhalf = 0.5f*x;
	int i = *(int *)&x;          // View x as an int.
	i = 0x5f37599e - (i >> 1);   // Initial guess.
	x = *(float *)&i;            // View i as float.
	x = x*(1.5f - xhalf*x*x);    // Newton step.
	x = x*(1.5f - xhalf*x*x);    // Newton step again.
	return x;
}

inline float accurateSqrt(float x)
{
	return x * rsqrt1(x);
}

inline void condSwap(bool c, float &X, float &Y)
{
	// used in step 2
	float Z = X;
	X = c ? Y : X;
	Y = c ? Z : Y;
}

inline void condNegSwap(bool c, float &X, float &Y)
{
	// used in step 2 and 3
	float Z = -X;
	X = c ? Y : X;
	Y = c ? Z : Y;
}

inline void approximateGivensQuaternion(float a11, float a12, float a22, float &ch, float &sh)
{
	/*
	* Given givens angle computed by approximateGivensAngles,
	* compute the corresponding rotation quaternion.
	*/
	ch = 2 * (a11 - a22);
	sh = a12;
	bool b = _gamma*sh*sh < ch*ch;
	// fast rsqrt function suffices
	// rsqrt2 (https://code.google.com/p/lppython/source/browse/algorithm/HDcode/newCode/rsqrt.c?r=26)
	// is even faster but results in too much error
	float w = accurateSqrt(ch*ch + sh*sh);
	ch = b ? w*ch : (float)_cstar;
	sh = b ? w*sh : (float)_sstar;
}

inline void jacobiConjugation(const int x, const int y, const int z,
	float &s11,
	float &s21, float &s22,
	float &s31, float &s32, float &s33,
	glm::quat qV)
{
	float ch, sh;
	approximateGivensQuaternion(s11, s21, s22, ch, sh);

	float scale = ch*ch + sh*sh;
	float a = (ch*ch - sh*sh) / scale;
	float b = (2 * sh*ch) / scale;

	// make temp copy of S
	float _s11 = s11;
	float _s21 = s21; float _s22 = s22;
	float _s31 = s31; float _s32 = s32; float _s33 = s33;

	// perform conjugation S = Q'*S*Q
	// Q already implicitly solved from a, b
	s11 = a*(a*_s11 + b*_s21) + b*(a*_s21 + b*_s22);
	s21 = a*(-b*_s11 + a*_s21) + b*(-b*_s21 + a*_s22);	s22 = -b*(-b*_s11 + a*_s21) + a*(-b*_s21 + a*_s22);
	s31 = a*_s31 + b*_s32;								s32 = -b*_s31 + a*_s32; s33 = _s33;

	// update cumulative rotation qV
	float tmp[3];
	tmp[0] = qV[0] * sh;
	tmp[1] = qV[1] * sh;
	tmp[2] = qV[2] * sh;
	sh *= qV[3];

	qV[0] *= ch;
	qV[1] *= ch;
	qV[2] *= ch;
	qV[3] *= ch;

	// (x,y,z) corresponds to ((0,1,2),(1,2,0),(2,0,1))
	// for (p,q) = ((0,1),(1,2),(0,2))
	qV[z] += sh;
	qV[3] -= tmp[z]; // w
	qV[x] += tmp[y];
	qV[y] -= tmp[x];

	// re-arrange matrix for next iteration
	_s11 = s22;
	_s21 = s32; _s22 = s33;
	_s31 = s21; _s32 = s31; _s33 = s11;
	s11 = _s11;
	s21 = _s21; s22 = _s22;
	s31 = _s31; s32 = _s32; s33 = _s33;

}

inline float dist2(float x, float y, float z)
{
	return x*x + y*y + z*z;
}

// finds transformation that diagonalizes a symmetric matrix
inline void jacobiEigenanlysis(glm::mat3 &S, glm::quat qV)
{
	for (int i = 0; i<4; i++)
	{
		// we wish to eliminate the maximum off-diagonal element
		// on every iteration, but cycling over all 3 possible rotations
		// in fixed order (p,q) = (1,2) , (2,3), (1,3) still retains
		//  asymptotic convergence
		jacobiConjugation(0, 1, 2, S[0][0], S[1][0], S[1][1], S[2][0], S[2][1], S[2][2], qV); // p,q = 0,1
		jacobiConjugation(1, 2, 0, S[0][0], S[1][0], S[1][1], S[2][0], S[2][1], S[2][2], qV); // p,q = 1,2
		jacobiConjugation(2, 0, 1, S[0][0], S[1][0], S[1][1], S[2][0], S[2][1], S[2][2], qV); // p,q = 0,2
	}
}


inline void sortSingularValues(glm::mat3 &B, glm::mat3 &V)
{
	float rho1 = dist2(B[0][0], B[1][0], B[2][0]);
	float rho2 = dist2(B[0][1], B[1][1], B[2][1]);
	float rho3 = dist2(B[0][2], B[1][2], B[2][2]);
	bool c;
	c = rho1 < rho2;
	condNegSwap(c, B[0][0], B[0][1]); condNegSwap(c, V[0][0], V[0][1]);
	condNegSwap(c, B[1][0], B[1][1]); condNegSwap(c, V[1][0], V[1][1]);
	condNegSwap(c, B[2][0], B[2][1]); condNegSwap(c, V[2][0], V[2][1]);
	condSwap(c, rho1, rho2);
	c = rho1 < rho3;
	condNegSwap(c, B[0][0], B[0][2]); condNegSwap(c, V[0][0], V[0][2]);
	condNegSwap(c, B[1][0], B[1][2]); condNegSwap(c, V[1][0], V[1][2]);
	condNegSwap(c, B[2][0], B[2][2]); condNegSwap(c, V[2][0], V[2][2]);
	condSwap(c, rho1, rho3);
	c = rho2 < rho3;
	condNegSwap(c, B[0][1], B[0][2]); condNegSwap(c, V[0][1], V[0][2]);
	condNegSwap(c, B[1][1], B[1][2]); condNegSwap(c, V[1][1], V[1][2]);
	condNegSwap(c, B[2][1], B[2][2]); condNegSwap(c, V[2][1], V[2][2]);
}


void QRGivensQuaternion(float a1, float a2, float &ch, float &sh)
{
	// a1 = pivot point on diagonal
	// a2 = lower triangular entry we want to annihilate
	float epsilon = (float)EPSILON;
	float rho = accurateSqrt(a1*a1 + a2*a2);

	sh = rho > epsilon ? a2 : 0;
	ch = fabsf(a1) + fmaxf(rho, epsilon);
	bool b = a1 < 0;
	condSwap(b, sh, ch);
	float w = accurateSqrt(ch*ch + sh*sh);
	ch *= w;
	sh *= w;
}


inline void QRDecomposition(glm::mat3 B, glm::mat3 &U, glm::mat3 &S)
{
	// matrix that we want to decompose
	float b11 = B[0][0], b12 = B[0][1], b13 = B[0][2];
	float b21 = B[1][0], b22 = B[1][1], b23 = B[1][2];
	float b31 = B[2][0], b32 = B[2][1], b33 = B[2][2];
	// output Q
	float &q11 = U[0][0], &q12 = U[0][1], &q13 = U[0][2];
	float &q21 = U[1][0], &q22 = U[1][1], &q23 = U[1][2];
	float &q31 = U[2][0], &q32 = U[2][1], &q33 = U[2][2];
	// output R
	float &r11 = S[0][0], &r12 = S[0][1], &r13 = S[0][2];
	float &r21 = S[1][0], &r22 = S[1][1], &r23 = S[1][2];
	float &r31 = S[2][0], &r32 = S[2][1], &r33 = S[2][2];

	float ch1, sh1, ch2, sh2, ch3, sh3;
	float a, b;

	// first givens rotation (ch,0,0,sh)
	QRGivensQuaternion(b11, b21, ch1, sh1);
	a = 1 - 2 * sh1*sh1;
	b = 2 * ch1*sh1;
	// apply B = Q' * B
	r11 = a*b11 + b*b21;  r12 = a*b12 + b*b22;  r13 = a*b13 + b*b23;
	r21 = -b*b11 + a*b21; r22 = -b*b12 + a*b22; r23 = -b*b13 + a*b23;
	r31 = b31;          r32 = b32;          r33 = b33;

	// second givens rotation (ch,0,-sh,0)
	QRGivensQuaternion(r11, r31, ch2, sh2);
	a = 1 - 2 * sh2*sh2;
	b = 2 * ch2*sh2;
	// apply B = Q' * B;
	b11 = a*r11 + b*r31;  b12 = a*r12 + b*r32;  b13 = a*r13 + b*r33;
	b21 = r21;           b22 = r22;           b23 = r23;
	b31 = -b*r11 + a*r31; b32 = -b*r12 + a*r32; b33 = -b*r13 + a*r33;

	// third givens rotation (ch,sh,0,0)
	QRGivensQuaternion(b22, b32, ch3, sh3);
	a = 1 - 2 * sh3*sh3;
	b = 2 * ch3*sh3;
	// R is now set to desired value
	r11 = b11;             r12 = b12;           r13 = b13;
	r21 = a*b21 + b*b31;     r22 = a*b22 + b*b32;   r23 = a*b23 + b*b33;
	r31 = -b*b21 + a*b31;    r32 = -b*b22 + a*b32;  r33 = -b*b23 + a*b33;

	// construct the cumulative rotation Q=Q1 * Q2 * Q3
	// the number of floating point operations for three quaternion multiplications
	// is more or less comparable to the explicit form of the joined matrix.
	// certainly more memory-efficient!
	float sh12 = sh1*sh1;
	float sh22 = sh2*sh2;
	float sh32 = sh3*sh3;

	q11 = (-1 + 2 * sh12)*(-1 + 2 * sh22);
	q12 = 4 * ch2*ch3*(-1 + 2 * sh12)*sh2*sh3 + 2 * ch1*sh1*(-1 + 2 * sh32);
	q13 = 4 * ch1*ch3*sh1*sh3 - 2 * ch2*(-1 + 2 * sh12)*sh2*(-1 + 2 * sh32);

	q21 = 2 * ch1*sh1*(1 - 2 * sh22);
	q22 = -8 * ch1*ch2*ch3*sh1*sh2*sh3 + (-1 + 2 * sh12)*(-1 + 2 * sh32);
	q23 = -2 * ch3*sh3 + 4 * sh1*(ch3*sh1*sh3 + ch1*ch2*sh2*(-1 + 2 * sh32));

	q31 = 2 * ch2*sh2;
	q32 = 2 * ch3*(1 - 2 * sh22)*sh3;
	q33 = (-1 + 2 * sh22)*(-1 + 2 * sh32);
}

void svd(glm::mat3 A, glm::mat3 &U, glm::mat3 &S, glm::mat3 &V)
{
	glm::mat3 ATA = glm::transpose(A) * A;

	// symmetric eigenalysis
	glm::quat qV(1, 0, 0, 0);
	jacobiEigenanlysis(ATA, qV);
	V = glm::toMat3(qV);

	glm::mat3 B = A * V;

	// sort singular values and find V
	sortSingularValues(B,V);

	// QR decomposition
	QRDecomposition(B,U,S);
}

#endif