#pragma once
typedef struct float2 {
  float x;
  float y;
} float2;

inline float2 make_float2(float x, float y) {
  float2 result; result.x = x; result.y = y; return result;
}

inline float2 operator*( float2 a, float2 b ) { return make_float2( a.x*b.x-a.y*b.y, a.x*b.y+a.y*b.x ); }
inline float2 operator+( float2 a, float2 b ) { return make_float2( a.x + b.x, a.y + b.y ); }
inline float2 operator-( float2 a, float2 b ) { return make_float2( a.x - b.x, a.y - b.y ); }
inline float2 operator*( float2 a, float b ) { return make_float2( b*a.x , b*a.y); }

