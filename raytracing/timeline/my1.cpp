#include <iostream>
#include <vector>
#include <algorithm>
#include <memory>
#include <thread>

#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#include "Bitmap.h"

/*#include <iostream>
#include <fstream>
#include <stdio.h>

#include <vector>
#include <cmath>
#include <algorithm>

#define EPS 0.0000001
#define PI 3.14159265358979323846
#define PI2 6.283185307179586232

using namespace std;

#define range(i, j, k) for (int i = j; i < k; i++)
typedef long long int ll;
typedef int gmint;
typedef long double gmfloat;
typedef gmint gmvec;
typedef gmvec gmline;
gmint INF = 1000 * 1000 * 1000;


struct Vec;
struct Line;
Line line_pd(const Vec&, const Vec&);
Line line_2p(const Vec&, const Vec&);
pair<int, Vec> line_intersect(const Line&, const Line&);
gmvec dpr(const Vec&, const Vec&);
gmvec cpr(const Vec&, const Vec&);

template<typename tnum>
tnum abs(tnum n)
{
    if (n < 0) return -n;
    return n;
}

template<typename tnum1, typename tnum2>
bool eq(tnum1 a, tnum2 b) {
    return abs(a - b) < EPS;
}


struct Vec {
    gmvec x, y;

    Vec(): x(0), y(0) {}

    Vec(gmvec x0, gmvec y0): x(x0), y(y0) {}

    gmfloat len2() {
        return x * x + y * y;
    }

    gmfloat len() {
        return sqrt(len2());
    }

    Vec operator + (const Vec &v) const {
        return Vec(x + v.x, y + v.y);
    }

    Vec operator - (const Vec &v) const {
        return Vec(x - v.x, y - v.y);
    }

    Vec operator - () const {
        return Vec(-x, -y);
    }

    Vec operator * (gmint k) const {
        return Vec(gmvec(x * k), gmvec(y * k));
    }

    Vec operator * (gmfloat k) const {
        return Vec(gmvec(x * k), gmvec(y * k));
    }

    Vec dir() {
        gmfloat d = len();
        return Vec(gmvec(x / d), gmvec(y / d));
    }

    bool operator < (const Vec &v) const {
        return (y < v.y && eq(x, v.x) && !eq(y, v.y)) || x < v.x;
    }

    bool operator > (const Vec &v) const {
        return (y > v.y && eq(x, v.x) && !eq(y, v.y)) || x > v.x;
    }

    bool operator == (const Vec &v) const {
        return eq(x, v.x) && eq(y, v.y);
    }

    bool operator >= (const Vec &v) const {
        return (*this > v) || (*this == v);
    }

    bool operator <= (const Vec &v) const {
        return (*this < v) || (*this == v);
    }

    bool operator != (const Vec &v) const {
        return !(*this == v);
    }

    bool operator || (const Vec &v) const { // Parallel
        return eq(cpr(*this, v), 0);
    }

    void print() {
        cout << x << " " << y << endl;
    }
};

istream &operator >> (istream &in, Vec &a) {
    in >> a.x >> a.y;
    return in;
}

ostream &operator << (ostream &out, const Vec &a) {
    out << a.x << " " << a.y << "\n";
    return out;
}

struct Line {
    Vec A, B;
    gmline a, b, c;

    Line() {
        a = 0; b = 0; c = 0;
    }

    Line(gmline x, gmline y, gmline z) {
        a = x; b = y; c = z;
        // Two points on the line
        if (x != 0) {
            A = Vec(-z / x, 0);
        }
        if (y != 0) {
            A = Vec(0, -z / y);
        }
        B = A + dir();
    }

    Vec norm() const {
        return Vec(gmvec(a), gmvec(b));
    }

    Vec dir() const {
        return Vec(-gmvec(b), gmvec(a));
    }

    Vec v_norm(const Vec v) const {
        return line_intersect(*this, line_pd(v, norm())).second;
    }

    bool contains(const Vec &v) const {
        return eq(v.x * a + v.y * b, -c);
    }

    bool operator == (const Line l) const {
        return eq(a * l.b, b * l.a) && eq(b * l.c, c * l.b) && eq(a * l.c, c * l.a);
    }
};

istream &operator >> (istream &in, Line &l) {
    in >> l.a >> l.b >> l.c;
    l = Line(l.a, l.b, l.c);
    return in;
}

ostream &operator << (ostream &out, const Line &l) {
    out << l.a << " " << l.b << " " << l.c << "\n";
    return out;
}

Line line_pd(const Vec &p, const Vec &d) {
    if (d != Vec(0, 0)) {
        return Line(d.y, -d.x, d.x * p.y - d.y * p.x);
    }
    return Line();
}

Line line_2p(const Vec &v1, const Vec &v2) {
    if (!(v1 == v2)) {
        return Line(v2.y - v1.y, v1.x - v2.x, v1.y * (v2.x - v1.x) - v1.x * (v2.y - v1.y));
    }
    return Line();
}

// Line intersection: returns <true, Vec> if there's 1 point; otherwise returns <false, Vec>:
// Vec == Vec(0, 0) if none, Vec == Vec(1, 1) if infinity.
pair<bool, Vec> l_intersect(const Line &l1, const Line &l2)
{
    if (l1 == l2) {
        return make_pair(false, Vec(1, 1));
    } else if (l1.a * l2.b == l1.b * l2.a) {
        return make_pair(false, Vec(0, 0));
    } else {
        gmvec x = gmvec((-l2.b * l1.c + l1.b * l2.c) / (l1.a * l2.b - l2.a * l1.b));
        gmvec y = gmvec((-l1.a * l2.c + l2.a * l1.c) / (l1.a * l2.b - l2.a * l1.b));
        return make_pair(true, Vec(x, y));
    }
}

// Segment
struct Seg {
    Line l;
    Vec p1, p2;

    Seg(const Vec &v1, const Vec &v2) {
        l = line_2p(v1, v2);
        if (v1 <= v2) {
            p1 = Vec(v1.x, v1.y);
            p2 = Vec(v2.x, v2.y);
        } else {
            p2 = Vec(v1.x, v1.y);
            p1 = Vec(v2.x, v2.y);
        }
    }

    bool env(const Vec &v) const { // Envelopes
        return p1 <= v && v <= p2;
    }

    Vec divide(gmfloat a, gmfloat b) { // Returns the point that divides the given segment
        return p1 + ((p2 - p1) * (a / (a + b)));
    }
};

// Intersection of two segments
// Returns either an empty vector, one intersection point, or two points defining the intersection segment
vector<Vec> intersect_segments(Vec a, Vec b, Vec c, Vec d) {
    vector<Vec> res;
    if (a > b) {
        swap(a, b);
    }
    if (c > d) {
        swap(c, d);
    }
    Vec d_1 = b - a;
    Vec d_2 = d - c;
    if (cpr(d_1, d_2) != 0) {
        gmfloat t_1 = cpr((c - a), d_2) / cpr(d_1, d_2);
        gmfloat t_2 = cpr((a - c), d_1) / cpr(d_2, d_1);
        if ((0 <= t_1) && (t_1 <= 1) && (0 <= t_2) && (t_2 <= 1)) {
            res.push_back(a + (d_1 * t_1));
        }
    }
    else if (cpr((c - a), d_1) == 0 && cpr((c - a), d_2) == 0) {
        if (a > c) {
            swap(a, c);
            swap(d_1, d_2);
        }
        if (c + d_2 < a + d_1) {
            res.push_back(c);
            if (c != c + d_2) {
                res.push_back(c + d_2);
            }
        }
        else if (a + d_1 >= c) {
            res.push_back(c);
            if (a + d_1 != c) {
                res.push_back(a + d_1);
            }
        }
    }
    return res;
}

bool v_cmp(Vec a, Vec b) {
    return cpr(a, b) > 0 || (eq(cpr(a, b), 0) && a.len2() < b.len2());
}

vector<Vec> convex_hull(vector<Vec> point_set) {
    int N = point_set.size();
    vector<Vec> polygon(N);
    for (int i = 0; i < N; i++) {
        polygon[i] = point_set[i];
    }

    // Choose a starting point (lowest of left-most points)
    Vec A = point_set[0];
    for (int i = 0; i < N; i++) {
        Vec X = point_set[i];
        if (X.x < A.x || (X.x == A.x && X.y < A.y)) {
            A = X;
        }
    }

    // Sort all diagonal vectors
    vector<Vec> diagonals;
    for (int i = 0; i < N; i++) {
        if (!(point_set[i] == A)) {
            diagonals.push_back(-A + polygon[i]);
        }
    }

    sort(diagonals.begin(), diagonals.end(), v_cmp);

    // Construct the hull using a stack
    vector<Vec> hull;
    hull.push_back(Vec(0, 0));
    for (int i = 0; i < diagonals.size(); i++) {
        if (hull.size() > 1) {
            int last = hull.size() - 1;
            double angle = cpr(-hull[last - 1] + hull[last], -hull[last] + diagonals[i]);
            if (eq(angle, 0)) {
                hull.pop_back();
            }
            while (!eq(angle, 0) && angle < 0) {
                hull.pop_back();
                last = hull.size() - 1;
                angle = cpr(-hull[last - 1] + hull[last], -hull[last] + diagonals[i]);
            }
        }
        hull.push_back(diagonals[i]);
    }
    for (int i = 0; i < hull.size(); i++) {
        hull[i] = hull[i] + A;
    }

    return hull;
}

bool in_polygon(const vector<Vec> &pol, const Vec &v)
{
    int average = 0; // Average between 0 and 1
    int steps = 10;
    for (int i = 0; i < steps; i++) {
        Vec distant(INF + abs(rand()) * INF, INF + abs(rand()) * INF);
        Seg line = Seg(v, distant);
        int cnt = 0;
        for (int j = 0; j < pol.size(); j++) {
            Vec a, b;
            a = pol[j];
            if (j == pol.size() - 1) {
                b = pol[0];
            } else {
                b = pol[j + 1];
            }

            Seg arc(a, b);
            vector<Vec> res = intersect_segments(line.p1, line.p2, arc.p1, arc.p2);
            if (res.size() > 0) {
                cnt++;
            }
        }
        average += cnt % 2;
    }
    return average / steps;
}

// Dot(scalar) and cross(vector) product
gmvec dpr(const Vec &v1, const Vec &v2) // (a, b) = x1 * x2 + y1 * y2
{
    return v1.x * v2.x + v1.y * v2.y;
}

gmvec cpr(const Vec &v1, const Vec &v2) // [a, b] = x1 * y2 - y1 * x2
{
    return v1.x * v2.y - v1.y * v2.x;
}

// Making angles universal - from 0 to 2pi
gmfloat uni(gmfloat angle) {
    while (angle < 0) {
        angle += PI2;
    }
    while (angle >= PI2) {
        angle -= PI2;
    }
    return angle;
}

// Calculate the angle between vectors
gmfloat at2(Vec a, Vec b) {
    gmfloat res = atan2l(cpr(a, b), dpr(a, b));
    res = -res;
    return uni(res);
}

int main()
{
    cout.precision(16);
    cout << fixed;

    Vec A, B, C, D;
    cin >> A.x >> A.y >> B.x >> B.y >> C.x >> C.y >> D.x >> D.y;
    intersect_segments(A, B, C, D);


    return 0;
}*/


using std::vector;
using std::shared_ptr;
using std::make_shared;

const float INF = 1e9;

class Image {
	const int COLOR_DEPTH = 3;
	unsigned char *image;
	int n_rows, n_cols, n_colors;

	unsigned int *buf;

public:
	Image(int rows, int cols) {
		printf("Create\n");
		image = new unsigned char[rows * cols * COLOR_DEPTH];
		n_rows = rows;
		n_cols = cols;
		n_colors = COLOR_DEPTH;
	}

	unsigned char& operator()(int row, int col, int color) {
		return image[row * (n_cols * n_colors) + col * n_colors + color];
	}

	void save(const char *path) {
		buf = new unsigned int[n_rows * n_cols];
		int idx = 0;
		for (int y = n_rows - 1; y >= 0; y--) {
			for (int x = 0; x < n_cols; x++) {
				buf[y * n_cols + x] = (image[idx + 2] << 16) + (image[idx + 1] << 8) + image[idx];
				//printf("%u %u %u %u\n", buf[y * n_cols + x], image[idx], image[idx+1], image[idx+2]);

				idx += n_colors;
			}
		}

		SaveBMP(path, buf, n_cols, n_rows);
		delete buf;

		/*FILE *fout = fopen(path, "w");
		fprintf(fout, "P3\n%d %d\n255\n", n_cols, n_rows);
		int idx = 0;
		for (int y = n_rows - 1; y >= 0; y--) {
			for (int x = 0; x < n_cols; x++) {
				fprintf(fout, "%d %d %d\n",
						image[idx],
						image[idx + 1],
						image[idx + 2]);
				idx += n_colors;
			}
		}
		fclose(fout);*/
	}

	~Image() {
		printf("Destroy\n");
		delete image;
	}
};

class vec3 {
	float e[3];

public:
    vec3() {}
    vec3(float e0, float e1, float e2) { e[0] = e0; e[1] = e1; e[2] = e2; }
    inline float x() const { return e[0]; }
    inline float y() const { return e[1]; }
    inline float z() const { return e[2]; }
    inline float r() const { return e[0]; }
    inline float g() const { return e[1]; }
    inline float b() const { return e[2]; }

    inline const vec3& operator+() const { return *this; }
    inline vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }
    inline float operator[](int i) const { return e[i]; }
    inline float& operator[](int i) { return e[i]; }

    inline vec3& operator+=(const vec3 &v2);
    inline vec3& operator-=(const vec3 &v2);
    inline vec3& operator*=(const vec3 &v2);
    inline vec3& operator/=(const vec3 &v2);
    inline vec3& operator*=(const float t);
    inline vec3& operator/=(const float t);

    inline float length() const { return sqrt(e[0]*e[0] + e[1]*e[1] + e[2]*e[2]); }
    inline float squared_length() const { return e[0]*e[0] + e[1]*e[1] + e[2]*e[2]; }
    inline void make_unit_vector();

    friend inline vec3 operator+(const vec3 &v1, const vec3 &v2);
    friend inline vec3 operator-(const vec3 &v1, const vec3 &v2);
    friend inline vec3 operator*(const vec3 &v1, const vec3 &v2);
    friend inline vec3 operator/(const vec3 &v1, const vec3 &v2);
    friend inline vec3 operator*(float t, const vec3 &v);
    friend inline vec3 operator*(const vec3 &v, float t);
    friend inline vec3 operator/(const vec3 &v, float t);

    friend inline float dot(const vec3 &v1, const vec3 &v2);
    friend inline vec3 cross(const vec3 &v1, const vec3 &v2);

    friend inline std::istream& operator>>(std::istream &is, vec3 &t);
    friend inline std::ostream& operator<<(std::ostream &os, const vec3 &t);

};

inline std::istream& operator>>(std::istream &is, vec3 &t) {
    is >> t.e[0] >> t.e[1] >> t.e[2];
    return is;
}

inline std::ostream& operator<<(std::ostream &os, const vec3 &t) {
    os << t.e[0] << " " << t.e[1] << " " << t.e[2];
    return os;
}

inline void vec3::make_unit_vector() {
    float k = 1.0 / sqrt(e[0]*e[0] + e[1]*e[1] + e[2]*e[2]);
    e[0] *= k; e[1] *= k; e[2] *= k;
}

inline vec3 operator+(const vec3 &v1, const vec3 &v2) {
    return vec3(v1.e[0] + v2.e[0], v1.e[1] + v2.e[1], v1.e[2] + v2.e[2]);
}

inline vec3 operator-(const vec3 &v1, const vec3 &v2) {
    return vec3(v1.e[0] - v2.e[0], v1.e[1] - v2.e[1], v1.e[2] - v2.e[2]);
}

inline vec3 operator*(const vec3 &v1, const vec3 &v2) {
    return vec3(v1.e[0] * v2.e[0], v1.e[1] * v2.e[1], v1.e[2] * v2.e[2]);
}

inline vec3 operator*(float t, const vec3 &v) {
    return vec3(t*v.e[0], t*v.e[1], t*v.e[2]);
}

inline vec3 operator*(const vec3 &v, float t) {
    return vec3(t*v.e[0], t*v.e[1], t*v.e[2]);
}

inline vec3 operator/(const vec3 &v1, const vec3 &v2) {
    return vec3(v1.e[0] / v2.e[0], v1.e[1] / v2.e[1], v1.e[2] / v2.e[2]);
}

inline vec3 operator/(const vec3 &v, float t) {
    return vec3(v.e[0]/t, v.e[1]/t, v.e[2]/t);
}

inline float dot(const vec3 &v1, const vec3 &v2) {
    return v1.e[0]*v2.e[0]
         + v1.e[1]*v2.e[1]
         + v1.e[2]*v2.e[2];
}

inline vec3 cross(const vec3 &v1, const vec3 &v2) {
    return vec3(v1.e[1] * v2.e[2] - v1.e[2] * v2.e[1],
                v1.e[2] * v2.e[0] - v1.e[0] * v2.e[2],
                v1.e[0] * v2.e[1] - v1.e[1] * v2.e[0]);
}

inline vec3& vec3::operator+=(const vec3 &v) {
    e[0] += v.e[0];
    e[1] += v.e[1];
    e[2] += v.e[2];
    return *this;
}

inline vec3& vec3::operator-=(const vec3& v) {
    e[0] -= v.e[0];
    e[1] -= v.e[1];
    e[2] -= v.e[2];
    return *this;
}

inline vec3& vec3::operator*=(const vec3 &v) {
    e[0] *= v.e[0];
    e[1] *= v.e[1];
    e[2] *= v.e[2];
    return *this;
}

inline vec3& vec3::operator*=(const float t) {
    e[0] *= t;
    e[1] *= t;
    e[2] *= t;
    return *this;
}

inline vec3& vec3::operator/=(const vec3 &v) {
    e[0] /= v.e[0];
    e[1] /= v.e[1];
    e[2] /= v.e[2];
    return *this;
}

inline vec3& vec3::operator/=(const float t) {
    float k = 1.0/t;

    e[0] *= k;
    e[1] *= k;
    e[2] *= k;
    return *this;
}

inline vec3 unit_vector(vec3 v) {
    return v / v.length();
}



unsigned int module = 6700417;
unsigned int mul_k = 41;
unsigned int bias_c = 100001;
unsigned int cu_x = 204048;
float mxrand = module + 1.0;


inline float random_double() {
	//cu_x = (cu_x * mul_k + bias_c) % module;
	//return cu_x / mxrand;
	return (float) (random() / (RAND_MAX + 1.0));
}

vec3 random_in_unit_sphere() {
    /*vec3 p(0, 0, 0);
    do {
        p = 2 * vec3(random_double(), random_double(), random_double()) - vec3(1,1,1);
    } while (p.squared_length() >= 1.0);*/

	/*float phi = random_double() * 2 * M_PI;
    float psi = (random_double() - 0.5) * M_PI;
    float r = random_double();
    vec3 p = vec3(cos(phi), sin(phi), sin(psi)) * r;*/

    float phi = random_double() * 2 * M_PI;
    float z = random_double() * 2 - 1;
    float r = sqrt(1 - z * z);
    vec3 p = vec3(r * cos(phi), r * sin(phi), z);

    /*float y = random_double() * 4;
    float x;
    if (y < 2) {
		y -= 1;
		x = sqrt(1 - y * y);
	} else {
		y -= 3;
		x = -sqrt(1 - y * y);
	}
    float z = random_double() * 2 - 1;
    float r = sqrt(1 - z * z);
    vec3 p = vec3(r * x, r * y, z);*/

    return p;
}


class Ray {
	vec3 A;
	vec3 B;

public:
	Ray() {}
	Ray(const vec3 &a, const vec3 &b) { A = a; B = b; }
	vec3 origin() const       { return A; }
	vec3 direction() const    { return B; }
	vec3 point_at_parameter(float t) const { return A + t*B; }
};



class Camera {
	vec3 origin;
    float h_fov, v_fov;
	vec3 lower_left_corner;
	vec3 horizontal;
	vec3 vertical;

public:
	Camera(float horizontal_rad_fov=M_PI, float vertical_rad_fov=M_PI) {

		lower_left_corner = vec3(-2.0, -1.0, -1.0);
		horizontal = vec3(4.0, 0.0, 0.0);
		vertical = vec3(0.0, 2.0, 0.0);
		origin = vec3(0.0, 0.0, 0.0);
	}
	Ray get_ray(float u, float v) {
		return Ray(origin, lower_left_corner + u*horizontal + v*vertical - origin);
	}


};



class Material;
struct hit_record {
    float t;
    vec3 p;
    vec3 normal;
    vec3 color = vec3(-1, 0, 0);
    shared_ptr<Material> mat_ptr;
};

// Abstract class for rendering any surface or volume
class Hittable  {
public:
	virtual bool hit(
		const Ray &r, float t_min, float t_max, hit_record &rec) const = 0;
};

class HittableList: public Hittable {
	vector< shared_ptr<Hittable> > l; // Without pointer it won't compile

public:
	HittableList() {}
	HittableList(const vector< shared_ptr<Hittable> > &list) { l = list; }

	//vector< shared_ptr<Hittable> >& get_list() { return l; }

    void add(const shared_ptr<Hittable> &object) {
        l.push_back(object);
    }

	virtual bool hit(
		const Ray &r, float t_min, float t_max, hit_record& rec) const;
};

bool HittableList::hit(const Ray &r, float t_min, float t_max,
                        hit_record &rec) const {
    hit_record temp_rec;
    bool hit_anything = false;
    double closest_so_far = t_max;
    for (int i = 0; i < l.size(); i++) {
        if (l[i]->hit(r, t_min, closest_so_far, temp_rec)) {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }

    return hit_anything;
}



class Material {
public:
	/*
	 * On contact with surface, produce a new scattered ray.
	 * attenuation defines how much each color channel will be
	 * attenuated.
	 * Returns false if there will be no scattered ray (input
	 * ray was absorbed).
	 */
	virtual bool scatter(
		const Ray &r_in, const hit_record &rec, vec3 &attenuation, Ray &scattered
	) const = 0;
};

inline vec3 reflect(const vec3 &vec_in, const vec3 &normal) {
    return vec_in + 2 * normal * dot(vec_in, normal) * (-1);
}

class Lambertian : public Material {
    vec3 albedo;

public:
    Lambertian(const vec3 &a) { albedo = a; }

    virtual bool scatter(
        const Ray &r_in, const hit_record &hit_rec, vec3 &attenuation, Ray &scattered
    ) const {
        // attenuation and scattered are used as return values

        const vec3 normal = dot(r_in.direction(), hit_rec.normal) < 0 ?
                hit_rec.normal : -hit_rec.normal;
        vec3 scatter_direction = normal + random_in_unit_sphere();
        scattered = Ray(hit_rec.p, scatter_direction);
        attenuation = albedo;
        return true;
    }
};

class Metal : public Material {
    vec3 a;
    float f; // f is in [0, 1]

public:
    Metal(const vec3 &albedo, const float fuzziness) { a = albedo; f = fuzziness; }

    virtual bool scatter(
        const Ray &r_in, const hit_record &hit_rec, vec3 &attenuation, Ray &scattered
    ) const {
        // attenuation and scattered are used as return values

        const vec3 &vec_in = r_in.direction();
        float dot_v_n = dot(r_in.direction(), hit_rec.normal);
        const vec3 &normal = dot_v_n < 0 ? hit_rec.normal : -hit_rec.normal;
        float norm_h = fabs(dot_v_n);

        vec3 scatter_direction = vec_in + 2 * normal * norm_h +
                random_in_unit_sphere() * norm_h * f;

        scattered = Ray(hit_rec.p, scatter_direction);
        attenuation = a;
        return true;
    }
};

class Dielectric : public Material {
    vec3 a;
    float f; // f is in [0, 1]
    float th;
    float th_inv;
    float th_sq;
    float th_inv_sq;

public:
    Dielectric(const vec3 &albedo, const float fuzziness, const float theta) {
        a = albedo;
        f = fuzziness;
        th = theta;
        th_inv = 1.0 / theta;
        th_sq = theta * theta;
        th_inv_sq = 1.0 / (theta * theta);
    }

    virtual bool scatter(
        const Ray &r_in, const hit_record &hit_rec, vec3 &attenuation, Ray &scattered
    ) const {
        // attenuation and scattered are used as return values

        const vec3 &vec_in = r_in.direction();
        float dot_v_n = dot(r_in.direction(), hit_rec.normal);

        vec3 normal;
        float theta;
        float theta_sq;
        if (dot_v_n < 0) { // Ray enters body under surface
            normal = hit_rec.normal;
            theta = th;
            theta_sq = th_sq;
        } else {
            normal = -hit_rec.normal;
            theta = th_inv;
            theta_sq = th_inv_sq;
        }
        float norm_h = fabs(dot_v_n);

        vec3 u = vec_in + normal * norm_h;

        float cos_a = norm_h / vec_in.length();
        float sin_a_sq = 1 - cos_a * cos_a;

        float r0 = (1 - theta) / (1 + theta); // Shlick formula
        r0 = r0 * r0;
        float reflect_probability = r0 + (1 - r0) * pow(1 - cos_a, 5);

        if (sin_a_sq >= theta_sq || random_double() < reflect_probability) {
            // Only reflect
            vec3 scatter_direction = vec_in + 2 * normal * norm_h +
                    random_in_unit_sphere() * norm_h * f;
            scattered = Ray(hit_rec.p, scatter_direction);
        } else {
            // Refract through surface
            float sin_a = sqrt(sin_a_sq);
            vec3 refracted = -normal * sqrt(theta_sq - sin_a_sq) +
                    unit_vector(u) * sin_a;
            refracted = refracted +
                    random_in_unit_sphere() * dot(refracted, normal) * f;

            scattered = Ray(hit_rec.p, refracted);
        }
        attenuation = a;
        return true;
    }
};



class Sphere: public Hittable {
	vec3 C;
	float r;
	shared_ptr<Material> mat_ptr;

public:
	Sphere() {}
	Sphere(
        const vec3 &center, const float radius, const shared_ptr<Material> material
    ) { C = center; r = radius; mat_ptr = material; mat_ptr = material; }

	vec3 center() const 	{ return C; }
	float radius() const 	{ return r; }
	bool hit(const Ray &ray, float t_min, float t_max, hit_record &rec) const {
		// t^2 (B, B) + 2t (B, A-C) + (A-C, A-C) - R^2 >= 0 - discriminant
		float a = dot(ray.direction(), ray.direction());
		float b = dot(ray.direction(), ray.origin() - C); // Ommited *2
		float c = dot(ray.origin() - C, ray.origin() - C) - r * r;
		float d = b * b - a * c;

		if (d > 0) {
			float x1 = (-b - sqrt(d)) / a;
			if (t_min < x1 && x1 < t_max) {
				rec.t = x1;
				rec.p = ray.point_at_parameter(x1);
				rec.normal = unit_vector((rec.p - C) / r);
                rec.mat_ptr = mat_ptr;
				return true;
			}

			float x2 = (-b + sqrt(d)) / a;
			if (t_min < x2 && x2 < t_max) {
				rec.t = x2;
				rec.p = ray.point_at_parameter(x2);
				rec.normal = unit_vector((rec.p - C) / r);
                rec.mat_ptr = mat_ptr;
				return true;
			}
		}

		return false;
	}
};

class InfinitePlane: public Hittable {
	vec3 p; // Origin point
	vec3 u, v; // Direction vectors
	vec3 normal;

	float a, b, c, d;
    shared_ptr<Material> mat_ptr;

public:
	InfinitePlane() {}
	InfinitePlane(
        const vec3 &point, const vec3 &dir1, const vec3 &dir2, shared_ptr<Material> material
    ) {
		p = point;
		u = dir1;
		v = dir2;

		float tx = v.z() * u.y() - v.y() * u.z();
		a = -u.y() * tx;
		float ty = u.x() * v.z() - u.z() * v.x();
		b = u.y() * ty;
		float tz = v.x() * u.y() - v.y() * u.x();
		c = u.y() * tz;
		d = (p.x() * u.y() - p.y() * u.x()) * tx -
		    (p.z() * u.y() - p.y() * u.z()) * tz;

		normal = unit_vector(vec3(tx, -ty, -tz));

		v = cross(normal, u);
        mat_ptr = material;
	}

	bool hit(const Ray &ray, float t_min, float t_max, hit_record &rec) const {
		// M = p + t_1*u + t_2*v = q + t_3*s =>
		// t_1*u + t_2*v - t_3*s = q - p
		vec3 s = -ray.direction();
		vec3 origin = ray.origin() - p;
		float mat[3][4] = {
			{u.x(), v.x(), s.x(), origin.x()},
			{u.y(), v.y(), s.y(), origin.y()},
			{u.z(), v.z(), s.z(), origin.z()}
		};
		float det_3 = mat[0][0] * (mat[1][1] * mat[2][3] - mat[1][3] * mat[2][1]) -
					  mat[0][1] * (mat[1][0] * mat[2][3] - mat[1][3] * mat[2][0]) +
					  mat[0][3] * (mat[1][0] * mat[2][1] - mat[1][1] * mat[2][0]);
		float det_all = mat[0][0] * (mat[1][1] * mat[2][2] - mat[1][2] * mat[2][1]) -
					    mat[0][1] * (mat[1][0] * mat[2][2] - mat[1][2] * mat[2][0]) +
					    mat[0][2] * (mat[1][0] * mat[2][1] - mat[1][1] * mat[2][0]);
		float t = det_3 / det_all;

		vec3 f = ray.origin() + t * (-s) - p;
		float det_1 = f.x() * v.z() - v.x() - f.z();
		float det_2 = u.x() * f.z() - f.x() * u.z();
		float det_all2 = u.x() * v.z() - v.x() * u.z();
		float t1 = det_1 / det_all2;
		float t2 = det_2 / det_all2;

		if (t_min < t && t < t_max) {
			rec.t = t;
			rec.p = ray.point_at_parameter(t);
			rec.normal = normal;
			rec.color = vec3(1, 1, 1) * (((unsigned(t1 * 50) / 10) % 2 + (unsigned(t2 * 50) / 10) % 2) % 2);
            rec.mat_ptr = mat_ptr;
			return true;
		}

		return false;
	}
};

class Triangle: public Hittable {
	vec3 p; // Origin point
	vec3 u, v; // Direction vectors
	vec3 normal;

	float a, b, c, d;
    shared_ptr<Material> mat_ptr;

public:
	Triangle() {}
	Triangle(
        const vec3 &A, const vec3 &B, const vec3 &C, shared_ptr<Material> material
    ) {
		// a, b, c - in counterclockwise order to camera
		
		vec3 point = B;
		vec3 dir1 = A - B;
		vec3 dir2 = C - B;
		p = point;
		u = dir1;
		v = dir2;

		float tx = v.z() * u.y() - v.y() * u.z();
		a = -u.y() * tx;
		float ty = u.x() * v.z() - u.z() * v.x();
		b = u.y() * ty;
		float tz = v.x() * u.y() - v.y() * u.x();
		c = u.y() * tz;
		d = (p.x() * u.y() - p.y() * u.x()) * tx -
		    (p.z() * u.y() - p.y() * u.z()) * tz;

		normal = unit_vector(vec3(tx, -ty, -tz));

		v = cross(normal, u);
        mat_ptr = material;
	}

	bool hit(const Ray &ray, float t_min, float t_max, hit_record &rec) const {
		// M = p + t_1*u + t_2*v = q + t_3*s =>
		// t_1*u + t_2*v - t_3*s = q - p
		vec3 s = -ray.direction();
		vec3 origin = ray.origin() - p;
		float mat[3][4] = {
			{u.x(), v.x(), s.x(), origin.x()},
			{u.y(), v.y(), s.y(), origin.y()},
			{u.z(), v.z(), s.z(), origin.z()}
		};
		float det_3 = mat[0][0] * (mat[1][1] * mat[2][3] - mat[1][3] * mat[2][1]) -
					  mat[0][1] * (mat[1][0] * mat[2][3] - mat[1][3] * mat[2][0]) +
					  mat[0][3] * (mat[1][0] * mat[2][1] - mat[1][1] * mat[2][0]);
		float det_all = mat[0][0] * (mat[1][1] * mat[2][2] - mat[1][2] * mat[2][1]) -
					    mat[0][1] * (mat[1][0] * mat[2][2] - mat[1][2] * mat[2][0]) +
					    mat[0][2] * (mat[1][0] * mat[2][1] - mat[1][1] * mat[2][0]);
		float t = det_3 / det_all;

		vec3 f = ray.origin() + t * (-s) - p;
		float det_1 = f.x() * v.z() - v.x() - f.z();
		float det_2 = u.x() * f.z() - f.x() * u.z();
		float det_all2 = u.x() * v.z() - v.x() * u.z();
		float t1 = det_1 / det_all2;
		float t2 = det_2 / det_all2;

		if (t1 > 0 && t2 > 0 && t1 < 1 - t2 &&
				t_min < t && t < t_max) {
			rec.t = t;
			rec.p = ray.point_at_parameter(t);
			rec.normal = normal;
            rec.mat_ptr = mat_ptr;
			return true;
		}

		return false;
	}
};

vec3 bkg_color(const Ray &r) {
    vec3 unit_direction = unit_vector(r.direction());
    float t = 0.5*(unit_direction.y() + 1.0);
    return (1-t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
}

int reflections = 0;

vec3 color(const Ray &ray, const HittableList &objects, int recursion) {
    if (recursion > 20) {
		//printf("Exceeded\n");
		return vec3(0, 0, 0);
	}

	vec3 col;
	hit_record hit_rec;

	if (objects.hit(ray, 0.0001, INF, hit_rec)) {
		if (hit_rec.color[0] >= 0) { // Force a specific color
			col = hit_rec.color;
		} else { // Color-code the normal to surface
			vec3 N = hit_rec.normal;
			col = 0.5*vec3(N.x()+1, N.y()+1, N.z()+1);
		}
        vec3 attenuation;
        Ray scattered;
		if (hit_rec.mat_ptr->scatter(ray, hit_rec, attenuation, scattered)) {
            col = color(scattered, objects, recursion + 1) * attenuation;
    		reflections++;
        } else { // Ray absorbed :(
            col = vec3(0, 0, 0);
        }
	} else {
		col = bkg_color(ray);
	}
	return col;
}

float med(vector<float> v) {
	std::nth_element(v.begin(), v.begin() + v.size()/2, v.end());
	return v[v.size()/2];
}


void ray_caster(
    Image &img, int row_begin, int row_end,
    HittableList &objects, Camera &main_camera, int n, int m, int antialias)
{
    for (int i = row_begin; i < row_end; i++) {
		for (int j = 0; j < m; j++) {
			vec3 col(0, 0, 0);
			for (int it = 0; it < antialias; it++) {
				float v = (float(n - i) + random_double()) / float(n);
				float u = (float(j) + random_double()) / float(m);
				Ray r = main_camera.get_ray(u, v);
				col += color(r, objects, 0);
			}

			col /= (float) antialias;

			if (col[0] < 0.0 || col[0] > 1.0 ||
					col[1] < 0.0 || col[1] > 1.0 ||
					col[2] < 0.0 || col[2] > 1.0) {
				printf("Color out of [0.0, 1.0] range! Exiting...\n");
				exit(0);
			}

			img(i, j, 0) = sqrt(col[0]) * 255;
			img(i, j, 1) = sqrt(col[1]) * 255;
			img(i, j, 2) = sqrt(col[2]) * 255;
		}
	}
}

using std::thread;

int main() {
	int invokes = 0;

	int n = 1000*1/1;
	int m = 2000*1/1;
	int antialias = 3;
	Image img(n, m);

    /*vec3 **tmp_img = new vec3*[n];
    for (int i = 0; i < n; i++) {
        tmp_img[i] = new vec3[m];
    }*/


	Camera main_camera;

	HittableList objects;
    //objects.add(make_shared<Sphere>(
    //    vec3(0, 0, -1), 0.4, make_shared<Lambertian>(vec3(0.5, 0.6, 0.5))
    //));
    objects.add(make_shared<Sphere>(
        vec3(0, 0, -1), 0.1, make_shared<Lambertian>(vec3(1.0, 1.0, 0.0))
    ));
    objects.add(make_shared<Sphere>(
        vec3(0, 0, -1), 0.6, make_shared<Dielectric>(vec3(1.0, 1.0, 1.0), 0.01, 2.4)
    ));

    objects.add(make_shared<Sphere>(
        vec3(0.2, 0, -1), 0.1, make_shared<Dielectric>(vec3(1.0, 1.0, 1.0), 0, 1.5)
    ));
    objects.add(make_shared<Sphere>(
        vec3(-0.2, 0, -1), 0.1, make_shared<Dielectric>(vec3(1.0, 1.0, 1.0), 0, 1.5)
    ));

    objects.add(make_shared<Sphere>(
        vec3(0.8, 0, -1), 0.4, make_shared<Metal>(vec3(0.6, 1.0, 1.0), 0.1)
    ));
    /*objects.get_list().push_back(new Sphere(vec3(0.1, 0, -1.5), 0.6));
    objects.get_list().push_back(new Sphere(vec3(-0.5, 0, -1.5), 0.3));
    objects.get_list().push_back(new Sphere(vec3(-0.8, 0, -1.5), 0.1));
    objects.get_list().push_back(new Sphere(vec3(-1.5, 0, -1.5), 0.05));*/
    objects.add(make_shared<Sphere>(
        vec3(0, -100, 0), 99.5, make_shared<Lambertian>(vec3(0.8, 0.8, 0.01) * 0.5)
    ));
    objects.add(make_shared<Sphere>(
        vec3(-2, 1, -2), 1, make_shared<Metal>(vec3(0.8, 0.5, 0.5), 0)
    ));
    objects.add(make_shared<Sphere>(
        vec3(-0.8, 0, -1), 0.4, make_shared<Dielectric>(vec3(0.1, 1.0, 1.0), 0.01, 1.4)
    ));
    /*objects.add(make_shared<Sphere>(
        vec3(2, 0, -2), 1, make_shared<Dielectric>(vec3(0.1, 1.0, 1.0), 0, 1.4)
    ));*/
    /*objects.get_list().push_back(new InfinitePlane(
			vec3(0, -1, 0),
			vec3(1, -0.1, 0),
			vec3(0, 0, 1)
	));*/

    //Image &img, int row_begin, int row_end,
    //HittableList &objects, Camera &main_camera, int n, int m, int antialias)
	std::thread t1(ray_caster, std::ref(img), 0, n,
            std::ref(objects), std::ref(main_camera), n, m, antialias);
    //std::thread t2(ray_caster, std::ref(img), n / 2, n,
    //        std::ref(objects), std::ref(main_camera), n, m, antialias);
    t1.join();
    //t2.join();

	printf("%d %d\n", reflections, invokes);
	img.save("hi.bmp");

}
