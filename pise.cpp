// Ashutosh's Boiler Plate:-

#include <bits/stdc++.h>
using namespace std;

typedef long long int ll;
typedef vector<ll> vl;
typedef vector<vl> vvl;
typedef vector<bool> vb;
typedef pair<ll, ll> pll;
typedef map<ll, ll> mll;

const double PI = 3.141592653589793;
const int MOD = 1e9+7;
const int mod = 998244353;
const ll INF = LLONG_MAX;

#define fo(i, a, b, d) for(ll i = a; i < b; i += d)
#define of(i, a, b, d) for(ll i = a; i > b; i -= d)
#define fix(f,n) fixed<<setprecision(n)<<f
#define all(x) x.begin(), x.end()
#define cin std::cin
#define cout std::cout

ll modpower(ll x, ll y, ll m){
    x %= m;
    ll ans = 1;
    while (y > 0) {
        if (y % 2 == 1) ans = (ans*x) % m;
        x = (x*x) % m;
        y /= 2;
    }
    return ans;
}

bool customsort (const pll &a, const pll &b){
    return (a.first < b.first);
}

void solve(){
    
}

// for binary search
// ll l = ; // l is bad
// ll r = ; // r is good
// while (r > l+1){
//     ll mid = (r + l)/2;
//     if (good(mid)) r = mid;
//     else l = mid;
// }

int main(){
    
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr); cout.tie(nullptr);
    
    #ifndef ONLINE_JUDGE
        freopen("input.txt", "r", stdin);
        freopen("output.txt", "w", stdout);
    #endif

    // spf.assign(MAXN, 0);
    // sieve();

    ll t = 1;
    cin>>t;
    fo (i, 1, t+1, 1){
        // cout<<"Case #"<<i<<": ";
        solve();
    }

    return 0;
}

//Ashutosh's DSU Snippet:-

struct DSU{
    vl parent, mini, maxi, size;

    void init(ll n){
        parent.resize(n+1);
        mini.resize(n+1);
        maxi.resize(n+1);
        size.resize(n+1);
        fo (i, 1, n+1, 1) make(i);
    }

    void make(ll v){
        parent[v] = v;
        mini[v] = v;
        maxi[v] = v;
        size[v] = 1;
    }

    ll find(ll v){
        return parent[v] = v == parent[v] ? v : find(parent[v]);
    }

    void unite(ll x, ll y){
        x = find(x);
        y = find(y);
        if (x == y) return;
        if (size[x] < size[y]) swap(x, y);
        parent[y] = x;
        mini[x] = min(mini[x], mini[y]);
        maxi[x] = max(maxi[x], maxi[y]);
        size[x] += size[y];
    }
};

// Ashutosh's Combinatorics Snippet:-

ll modinv(ll n, ll p){
    return modpower(n, p - 2, p);
}

ll ncrmodp(ll n, ll r, ll p){
    if (n < r)
        return 0;
    if (r == 0)
        return 1;
    ll fac[n + 1];
    fac[0] = 1;
    fo (i, 1, n+1, 1)
        fac[i] = (fac[i - 1] * i) % p;
    return (fac[n]*modinv(fac[r], p)%p*modinv(fac[n - r], p)%p)%p;
}

// Ashutosh's SegTree Snippet:-

struct SegTree{
    ll size;
    vl tree_arr;
    vl lazy;

    ll combine(ll a, ll b) {return a+b;} //////////////////// Set min or max or add

    void initialize(vl &a){
        ll n = a.size();
        size = 1;
        while (size < n) size *= 2;
        tree_arr.resize(2*size);
        of (i, 2*size-1, -1, 1){
            if (i >= size+n-1)
                tree_arr[i] = 0; //////////////////// Set INF for min, -INF for max
            else if (i >= size-1)
                tree_arr[i] = a[i-size+1];
            else
                tree_arr[i] = combine(tree_arr[2*i+1], tree_arr[2*i+2]);
        }
        lazy.resize(2*size);
    }
    
    void set(ll ind, ll val, ll x, ll lx, ll rx){
        if (lx == rx - 1) {tree_arr[x] = val; return;}
        if (ind < (lx + rx)/2) set(ind, val, 2*x+1, lx, (lx + rx)/2);
        else set(ind, val, 2*x+2, (lx + rx)/2, rx);
        tree_arr[x] = combine(tree_arr[2*x+1], tree_arr[2*x+2]);
    }
    
    void set(ll ind, ll val){///////////////////////////// ind has 0 based indexing
        set(ind, val, 0, 0, size);
    }

    ll query(ll l, ll r, ll x, ll lx, ll rx){
        if (lx >= r || rx <= l) return 0;///////////////////// Set INF for min, -INF for max
        if (lx >= l && rx <= r) return tree_arr[x];
        ll query1 = query(l, r, 2*x+1, lx, (lx + rx)/2);
        ll query2 = query(l, r, 2*x+2, (lx + rx)/2, rx);
        return combine(query1, query2);
    }

    ll query(ll left, ll right){// answers query from left to right-1 (0 based indexing)
        return query(left, right, 0, 0, size);
    }

    void lazy_add(ll l, ll r, ll x, ll lx, ll rx, ll incr){
        tree_arr[x] += (rx-lx)*lazy[x];
        if (lx != rx-1){
            lazy[2*x+1] += lazy[x];
            lazy[2*x+2] += lazy[x];
        }
        lazy[x] = 0;
        if (lx >= r || rx <= l) return;
        if (lx >= l && rx <= r){
            tree_arr[x] += (rx-lx)*incr;
            if (lx != rx-1){
                lazy[2*x+1] += incr;
                lazy[2*x+2] += incr;
            }
            return;
        }
        lazy_add(l, r, 2*x+1, lx, (lx+rx)/2, incr);
        lazy_add(l, r, 2*x+2, (lx+rx)/2, rx, incr);
        tree_arr[x] = tree_arr[2*x+1] + tree_arr[2*x+2];
    }

    void lazy_add(ll left, ll right, ll incr){// adds incr to all elements from left to right-1 (0 based indexing)
        lazy_add(left, right, 0, 0, size, incr);
    }

    ll lazy_query(ll l, ll r, ll x, ll lx, ll rx){
        if (lx >= r || rx <= l) return 0;
        tree_arr[x] += (rx-lx)*lazy[x];
        if (lx != rx-1){
            lazy[2*x+1] += lazy[x];
            lazy[2*x+2] += lazy[x];
        }
        lazy[x] = 0;
        if (lx >= l && rx <= r) return tree_arr[x];
        ll query1 = lazy_query(l, r, 2*x+1, lx, (lx + rx)/2);
        ll query2 = lazy_query(l, r, 2*x+2, (lx + rx)/2, rx);
        return query1 + query2;
    }

    ll lazy_query(ll left, ll right){// gives sum of elements from left to right-1 (0 based indexing)
        return lazy_query(left, right, 0, 0, size);
    }
};

// Ashutosh's Sieve Snippet:-

ll MAXN = 1000000; // Consider taking MAXN = 31625 (Square root of 10^9)
vl spf;
vl primenums;
 
void sieve() {
    spf[1] = 1;
    fo (i, 2, MAXN, 1)
        spf[i] = i;
    fo (i, 4, MAXN, 2)
        spf[i] = 2;
    fo (i, 3, MAXN, 1)
        if (spf[i] == i)
            fo (j, i*i, MAXN, i)
                if (spf[j]==j)
                    spf[j] = i;
    fo (i, 2, MAXN, 1)
        if (spf[i] == i)
            primenums.push_back(i);
}
 
vector<pll> getFactorization(ll x){
    vector<pll> ans;
    while (x != 1){
        if (ans.size() == 0 || ans[ans.size()-1].first != spf[x])
            ans.push_back({spf[x], 1});
        else
            ans[ans.size()-1].second++;
        x = x / spf[x];
    }
    return ans;
}

