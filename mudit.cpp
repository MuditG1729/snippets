// Bellman ford
// pair<pair<int,int>,int> edge[m];
int s=1,u,v,wuv;
ll d[n+1];
fr(i,0,n+1,1){
    d[i] = INF;
}
d[s] = 0;
fr(i,1,n,1){
    fr(j,0,m,1){
        u = edge[j].first.first;
        v = edge[j].first.second;
        wuv = edge[j].second;
        if((d[u]<INF)&&(d[v]>(d[u]+wuv))){
            d[v] = d[u]+wuv;
        }
    }
}
// neg cycle checking
bool nc[n+1]={};    // reachable from neg-cycle
fr(i,0,n,1){
    fr(j,0,m,1){
        u = edge[j].first.first;
        v = edge[j].first.second;
        wuv = edge[j].second;
        if((d[u]<INF)&&(d[v]>(d[u]+wuv))){
            d[v] = d[u]+wuv;
            nc[v]=true;
        }
        else if(nc[u])nc[v]=true;
    }
}


// BFS
#include <queue>
int s=1,u;
int d[n+1];
int par[n+1];
fr(i,0,n+1,1){
    d[i] = -1;
    par[i] = i;
}
d[s] = 0;
queue<int> Q;
Q.push(s);
while(!Q.empty()){
    u = Q.front();
    Q.pop();
    for(int v:adj[u]){
        if(d[v]==-1){
            d[v] = d[u]+1;
            par[v] = u;
            Q.push(v);
        }
    }
}


// BigNum
struct BigNum{
    unsigned long long q,r;
    ll lim = 1e10l;

    BigNum():q(0ull),r(0ull){}
    BigNum(ll m):q(0ull),r(0ull),lim(m){}

    BigNum& operator += (ll x){
        this->r += (x%lim);
        this->q += (ll)(x/lim);
        this->q += (ll)(r/lim);
        this->r %= lim;
        return *this;
    }
    // Make sure x does not overflow!!
};
ostream & operator << (ostream &out, const BigNum &c){
    ll d = (c.r==0)?0:((ll)(log2(c.r)/log2(10))+1);
    d = 10-d;
    if(c.q>0){
        out << c.q;
        while(d--)out << 0;
    }
    out << c.r;
    return out;
}


// DFS
#include <vector>
// size = number of vertices
vector<int> adj[100001];
bool visited[100001]={};
 
void DFS_Visit(int u){
    for(int v:adj[u]){
        if(!visited[v]){
            visited[v] = true;
            DFS_Visit(v);
        }
    }
}

// Dijkstra
#include <vector>
#include <priority_queue>
// make sure there are no parallel edges.
ll s=1,u,du;
ll d[n+1];
fr(i,0,n+1,1){
    d[i]=INF;
}
d[s]=0;
priority_queue<pair<ll,ll>, vector<pair<ll,ll>>, greater<pair<ll,ll>>> PQ;
fr(i,1,n,1){
    PQ.push({d[i],i});
}
while(!PQ.empty()){
    du = PQ.top().first;
    u = PQ.top().second;
    PQ.pop();
    
    if(d[u]<du)continue;
    if(du==INF)continue;// unreachable

    for(pair<ll,ll> p:adj[u]){
        if(d[p.second]>d[u]+(p.first)){
            d[p.second] = d[u]+p.first;
            PQ.push({d[p.second],p.second});
        }
    }
}


// DSU
// size = number of elements in total
ll parent[2502]={};
ll siz[2502]={};
int find_set(int v) {
    if (v == parent[v])
        return v;
    return parent[v] = find_set(parent[v]);
}
void make_set(int v) {
    parent[v] = v;
    siz[v] = 1;
}
void union_sets(int a, int b) {
    a = find_set(a);
    b = find_set(b);
    if (a != b) {
        if (siz[a] < siz[b])
            swap(a, b);
        parent[b] = a;
        siz[a] += siz[b];
    }
}

// Lagrange interpolation
// f(x) = SUM( f(i) * PRODUCT((x-xj) / (xj-xi)) )    sum is over i, product is over j
modfac(MAXD,mod);    //precompute this, MAXD=maximum degree
ll d = k+1;    //degree of polynomial
ll p = d+1;    // number of data-points
ll x = n;    // point at which value is required
ll f[p+1]={};    // value at 1,2,3, . . ., p
f[0] = 0;    // dummy
fr(i,1,p+1,1){
    // initiate all f[i]
    f[i] = (f[i-1]+modpower(i,k,mod))%mod;
}
if(x<p+1)cout << f[x] << endl;
else{
    ll ans=0;
    ll pre[p+1],suf[p+1];
    pre[1]=1;suf[p]=1;
    fr(i,2,p+1,1){
        pre[i] = pre[i-1]*(x-i+1);
        pre[i] %= mod;
        suf[p+1-i] = suf[p+2-i]*(x-(p+2-i));
        suf[p+1-i] %= mod;
    }
    int parity=(p&1)?1:(-1);
    fr(i,1,p+1,1){
        // change below line according to problem
        ans += (ll)(((((f[i]*pre[i])%mod)*invfac[p-i])%mod)*((invfac[i-1]*suf[i])%mod)%mod)*parity;
        ans = maud(ans,mod);
        parity = -parity;
    }
    cout << ans << endl;
}


// LCA
vector<int> adj[MAXN];
bool visited[MAXN]={};
int euler[N] = {};
int first[N] = {};// first occurence in euler
int height[N] = {};
int ind=0;

int sparse_f(int a, int b){
    return height[a] < height[b] ? a : b;
}
 
void DFS_Visit(int u){
    first[u] = ind;
    euler[ind++] = u;
    for(int v:adj[u]){
        if(!visited[v]){
            visited[v] = true;
            height[v] = height[u] + 1;
            DFS_Visit(v);
            euler[ind++] = u;
        }
    }
}

int lca(int u, int v){
    int left = first[u], right = first[v];
    if (left > right)
        swap(left, right);
    return q_sparse(left, right);
}
// call init_sparse(euler)



// Sniper
#include <iostream>
#include <climits>
// #include <string>
// #include <algorithm>
// #include <vector>
// #include <map>
// #include <unordered_map>
// #include <set>
// #include <unordered_set>
// #include <bits/stdc++.h>
// #include <cmath>

using namespace std;

typedef long long ll;
typedef long double ld;

const int inf = INT_MAX;
const ll INF = LLONG_MAX;
const double PI = 3.141592653589793;
const ll MOD = 998244353;
const ll mod = 1e9 + 7;

#define fr(i, a, b, d) for(ll i = a; i < b; i += d)
#define rf(i, a, b, d) for(ll i = a; i >= b; i -= d)
#define maud(x,m) (((x)%m+m)%m)

ll max(ll x, ll y){if(x>y) return x;return y;}
ll min(ll x, ll y){if(x<y) return x;return y;}
ll max(ll x, ll y, ll z){return max(x,max(y,z));}
ll min(ll x, ll y, ll z){return min(x,min(y,z));}

#define traverse(container, it)\
    for (auto it = container.begin(); it != container.end(); it++)
#define rtraverse(container, it)\
    for (auto it = container.rbegin(); it != container.rend(); it++)

#define pb push_back

#define MAXN 100001



// Try BINARY-SEARCH on answer !!!
// To return maximum answer, ALWAYS initialize answer to -INF-1, similarly for minimum INF.
// Check condition for while-loop (or vs and)

// declare any array of size~1e6 as global.
// string addition (concatination) is not O(1), avoid it.
// str.length() is unsigned int, cast is to int while subtracting it.
// Interger overflow susceptible problems: Dijkstra, continuous multiplication, 
// Try dynamic array instead of vector, and delete it properly


// Matrix
#include <vector>
#include <cassert>
struct Mat {
  int n, m;
  vector<vector<int>> a;
  Mat() { }
  Mat(int _n, int _m) {n = _n; m = _m; a.assign(n, vector<int>(m, 0)); }
  Mat(vector< vector<int> > v) { n = v.size(); m = n ? v[0].size() : 0; a = v; }
  inline void make_unit() {
    assert(n == m);
    for (int i = 0; i < n; i++)  {
      for (int j = 0; j < n; j++) a[i][j] = i == j;
    }
  }
  static Mat I(long long k){
      Mat ans(k,k);
      ans.make_unit();
      return ans;
  }
  inline Mat operator + (const Mat &b) {
    assert(n == b.n && m == b.m ||
        !(std::cerr << "Error: Cannot add matrices of sizes ["<< m << "," << n << "] and\
         ["<< b.n << "," << b.m << "]" << "\n"));
    Mat ans = Mat(n, m);
    for(int i = 0; i < n; i++) {
      for(int j = 0; j < m; j++) {
        ans.a[i][j] = (a[i][j] + b.a[i][j]) % mod;
      }
    }
    return ans;
  } 
  inline Mat operator - (const Mat &b) {
    assert(n == b.n && m == b.m ||
        !(std::cerr << "Error: Cannot subtract matrices of sizes ["<< m << "," << n << "] and\
         ["<< b.n << "," << b.m << "]" << "\n"));
    Mat ans = Mat(n, m);
    for(int i = 0; i < n; i++) {
      for(int j = 0; j < m; j++) {
        ans.a[i][j] = (a[i][j] - b.a[i][j] + mod) % mod;
      }
    }
    return ans;
  }
  inline Mat operator * (const Mat &b) {
    assert(m == b.n ||
        !(std::cerr << "Error: Cannot multiply matrices of sizes ["<< m << "," << n << "] and\
         ["<< b.n << "," << b.m << "]" << "\n"));
    Mat ans = Mat(n, b.m);
    for(int i = 0; i < n; i++) {
      for(int j = 0; j < b.m; j++) {
        for(int k = 0; k < m; k++) {
          ans.a[i][j] = (ans.a[i][j] + 1LL * a[i][k] * b.a[k][j] % mod) % mod;
        }
      }
    }
    return ans;
  }
  inline Mat pow(long long k) {
    assert(n == m ||
        !(std::cerr << "Error: Cannot exponent a matrix of size ["<< m << "," << n << "]\n"));
    Mat ans = I(n), t = a;
    while (k) {
      if (k & 1) ans = ans * t;
      t = t * t;
      k >>= 1;
    }
    return ans;
  }
  inline Mat& operator += (const Mat& b) { return *this = (*this) + b; }
  inline Mat& operator -= (const Mat& b) { return *this = (*this) - b; }
  inline Mat& operator *= (const Mat& b) { return *this = (*this) * b; }
  inline bool operator == (const Mat& b) { return a == b.a; }
  inline bool operator != (const Mat& b) { return a != b.a; }
};


// MaxPS
ll dp[1000001] = {};
 
ll maxps(string arr, ll l, ll r){
    // dp[i] = LENGTH of (maximum prefix which coincides with a suffix) for prefix of LENGTH i\
    w.r.t INDEX range (l,r), only l inclusive
    // size of dp = MAXN + 1
    dp[0]=-1;
    ll x=0;
    fr(i,2,r-l+1,1){
        x = dp[i-1];
        while(arr[x+l]!=arr[i+l-1]){
            x = dp[x];
            if(x==-1){break;}
        }
        dp[i] = x+1;
    }
    return dp[r-l];
}


// Number Theory
// Distinct values of floor(n/x)
// O(sqrt(n))
// for (int l = 1, r; l <= n; l = r + 1) {
//     r = n / (n / l);
//     //n / x yields the same value for l <= x <= r.
// }

#define MAXN 10000001

// O(MAXN log(MAXN))
// int mu[MAXN]={};
// void mobius() {
//     mu[1] = 1;
//     for (int i = 2; i < MAXN; i++){
//         mu[i]--;
//         for (int j = i + i; j < MAXN; j += i) {
//           mu[j] -= mu[i];
//         }
//     }
// }

// O(MAXN log(MAXN))
// ll f[MAXN]={};
// ll h[MAXN];
// void mobius_inversion(){
//     for (int i = 1; i < MAXN; i++) { 
//         for (int j = i; j < MAXN; j += i) {
//             f[j] += h[i] * mu[j/i];
//         } 
//     }
// }

// O(log(min(a,b)))
ll ggcd(ll a, ll b) {
    if (!a || !b)
        return a | b;
    unsigned shift = __builtin_ctz(a | b);
    a >>= __builtin_ctz(a);
    do {
        b >>= __builtin_ctz(b);
        if (a > b)
            swap(a, b);
        b -= a;
    } while (b);
    return a << shift;
}

// O(log(min(a,b)))
// take abs before passing!!
// ll extgcd(ll a, ll b, ll& x, ll& y) {
//     if (b == 0) {
//         x = 1;
//         y = 0;
//         return a;
//     }
//     ll x1, y1;
//     ll d = extgcd(b, a % b, x1, y1);
//     x = y1;
//     y = x1 - y1 * (a / b);
//     return d;
// }

// O(log(y))
// (x raised to y) mod m
// ll modpower(ll x, ll y, ll m){
//  x %= m;
//  ll ans = 1;
//  while (y > 0) {
//      if (y % 2 == 1) ans = (ans*x) % m;
//      x = (x*x) % m;
//      y /= 2;
//  }
//  return ans;
// }

// O(logm/loglogm)
// ll modinv(ll i,ll m) {
//     i %= m;
//   return i <= 1 ? i : m - (long long)(m/i) * modinv(m % i,m) % m;
// }

// Try to avoid modinv as it can increase time slightly (~1s)

// O(nlogm)
// ll fac[1000001];
// ll invfac[1000001];
// void modfac(ll n, ll m){
//  fac[0] = 1;
//  invfac[0] = modinv(1,mod);
//  for (int i = 1; i <= n; i++){
//      fac[i] = (fac[i - 1] * i) % m;
//       invfac[i] = modinv(fac[i],mod);
//  }
// }

// O(log(m))
// ll modncr(ll n, ll r, ll m){
//     if (n < r)
//         return 0;
//     if (r == 0)
//         return 1;
//     return (fac[n] * modinv(fac[r], m) % m* modinv(fac[n - r], m) % m)% m;
// }

// O(p + log(n))
// ll modncr_lucas(ll n, ll r, ll p){
//     if(n<r)return 0;
//     if(r==0)return 1;
//     return (modncr(n%p, r%p, p)*modncr_lucas((n/p), (r/p), p))%p;
// }

/* CRT:
    x = ai % mi
    compute M =m1*m2*m3...
    for all mi: M' = M/mi
                x += (ai * M' * modinv(M',mi))
                x %= M;
Extended CRT:
    mi are not pairwise coprime
    Solution exits only if: ai = aj % gcd(mi,mj) for all i,j
    One by one merge two equations: x = A' % lcm(m1,m2)
        A' = (a1*(m2/g)*q + a2*(m1/g)*p)
        compute p and q by: extgcd(m1/g, m2/g, &p, &q)
        g = gcd(m1,m2)
*/


// whattodo
/*
1. Find number of factors of n: O(sqrt(n)); normal brute till sqrt(n);
2. Find all factors of a number n: O(sqrt(n)); normal brute till sqrt(n);
3. Find sum of all factors of a number n: O(sqrt(n)); normal brute till sqrt(n);
4. Find number of factors of q numbers less than n: O(cbrt(n)*q) + P(nloglogn);\
    use isprime sieve, find list of all primes till n, do separately for primes less(& more) than cbrt(n).
5. Find all factors of q number less than n: O(log(n)*q) + P(nloglogn);\
    use spf, keep dividing by spf and find factor.
6. Find sum of all factors of q numbers less than n: O(cbrt(n)*q) + P(nloglogn);\
    same as (4.), use formula of sum of factors;
7. Find sum of all factors of all numbers less than n: O(sqrt(n))
    formula = sum(i*floor(n/i)); floor(n/i) takes apprx 2*sqrt(n) unique values.
8. Find number of prime-factors of n: O(sqrt(n))
*/

// nt-bounds
/*
Bounds:
    1. Number of factors of N ~ O(cbrt(N)); (less for N>1e9, more for N<=1e9)
    2. Number of prime-factors of N ~ O(logN/loglogN)
    3. Number of primes less than N ~ O(N/logN)
    4. Nth prime = W(NlogN)
    5. Max gap between primes ~ (<118 for p upto 1e6; <288 for p upto 1e9; <1500 for p upto 1e18)
*/


// SegTree
struct node{
    ll value;
    node(){value=0;}
    node(ll v){value=v;}
};

constexpr ll SIZE = (1<<18) - 1; //(>2*MAXN)
constexpr ll R = (1<<17); //(SIZE+1)/2
node sgt[SIZE];
node IDENTITY;

node merge(node n1, node n2){
    return node(n1.value + n2.value);
}

void init(){
    IDENTITY = node(0);
    fr(i,0,SIZE,1)sgt[i] = IDENTITY;
}

void build(ll* arr, ll n, ll x, ll lx, ll rx){
    if(rx - lx == 1){
        if(lx < n)
            sgt[x] = node(arr[lx]);
        return;
    }
    ll m = (rx+lx)/2;
    build(arr,n,2*x+1,lx,m);
    build(arr,n,2*x+2,m,rx);
    sgt[x] = merge(sgt[2*x+1], sgt[2*x+2]);
}

void update(ll i, ll v, ll x, ll lx, ll rx){
    if(rx-lx==1){
        sgt[x] = node(v);
        return;
    }
    ll m = (lx+rx)/2;
    if(i<m){
        update(i,v,2*x+1,lx,m);
    }
    else{
        update(i,v,2*x+2,m,rx);
    }
    sgt[x] = merge(sgt[2*x+1],sgt[2*x+2]);
}

node query(ll l, ll r, ll x, ll lx, ll rx){
    if((l>=rx)||(lx>=r))return IDENTITY;
    if((lx>=l)&&(rx<=r))return sgt[x];
    ll m = (lx+rx)/2;
    return merge(query(l,r,2*x+1,lx,m), query(l,r,2*x+2,m,rx));
}

// Sieve
#define MAXN 10000001
int spf[MAXN];

void sieve(){
    spf[1] = 1;
    for (int i = 2; i < MAXN; i++)
        spf[i] = i;
    for (int i = 4; i < MAXN; i += 2)
        spf[i] = 2;
    // run this till sqrt(MAXN) accordingly - slight optimization
    for (int i = 3; i < MAXN; i+=2) {
        if ((spf[i] == i)) {
            // this loop can also begin from j=i (slight difference)
            for (ll j = (ll)i * i; j < MAXN; j += i)
                if (spf[j] == j)
                    spf[j] = i;
        }
    }
}

// bool iscomp[MAXN];

// void sieve(){
//     for (int i = 4; i < MAXN; i += 2)
//         iscomp[i] = true;
//     // run this till sqrt(MAXN) accordingly - slight optimization
//     for (int i = 3; i < MAXN; i+=2) {
//         if ((!iscomp[i])) {
//             // this loop can also begin from j=i (slight difference)
//             for (ll j = (ll)i * i; j < MAXN; j += i)
//                 iscomp[j] = true;
//         }
//     }
// }

//things we can do:
// 1. Do some operation on all prime factors of a number(~1e7): find spf.
// 2. Do some operation on all primes less than n(~1e7): find bool isprime[].
// 3. Find number of factors of n in O(sqrt(n)): bool isprime[sqrt(n)], then find list of primes,\
    iterate thru all primes.

// O((R-L)logR + sqrt R)
void range_sieve(char* isprime, ll L, ll R){
    // size of isprime array = R-L+1
    for(ll i=0;i<(R-L+1);i++)
        isprime[i] = true;
    for (ll i = 2; i * i <= R; i++) {
        for (ll j = max( i*i, ((L+i-1)/i)*i ); j <= R; j += i)
            isprime[j-L] = false;
    }
    if(L==1)
        isprime[0] = false;
}


// Sparse
#define N 100001
#define K 17 // ceil(log2(N))

int sparse[K][N];

int sparse_f(int a, int b);

// O(NlogN)
void init_sparse(int* arr){
    // size of arr = N
    // sparse_f: function for query
    copy(arr,arr+N,sparse[0]);
    for(int i = 1; i < K; i++){
        for(int j = 0; j + (1 << i) < N; j++)
            sparse[i][j] = sparse_f(sparse[i - 1][j], sparse[i - 1][j + (1 << (i - 1))]);
    }
}

int q_sparse(int L, int R){
    // O(1) for RMQ or similar (idempotent):
    int i = log2_floor(R - L + 1);
    return sparse_f(sparse[i][L], sparse[i][R - (1 << i) + 1]);
    // O(K) for Range sum:
    long long sum = 0;
    for (int i = K-1; i >= 0; i--) {
        if ((1 << i) <= R - L + 1) {
            sum += sparse[i][L];
            L += 1 << i;
        }
    }
    return sum;
}


