int storm(
    int *w,
    int a,
    int b,
    int c,
    int d,
    int e,
    int f,
    int g,
    int h)
{
    for (int i = 0; i < 64; i++) {

        int s1 =
            (e >> 6 | e << 26) ^
            (e >> 11 | e << 21) ^
            (e >> 25 | e << 7);

        int ch =
            (e & f) ^ (~e & g);

        int t1 =
            h + s1 + ch + w[i];

        int s0 =
            (a >> 2 | a << 30) ^
            (a >> 13 | a << 19) ^
            (a >> 22 | a << 10);

        int maj =
            (a & b) ^ (a & c) ^ (b & c);

        int t2 =
            s0 + maj;

        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }

    return a ^ b ^ c ^ d ^ e ^ f ^ g ^ h;
}

int storm2(
    int *w,
    int a,
    int b,
    int c,
    int d,
    int e,
    int f,
    int g,
    int h)
{
    for (int i = 0; i < 64; i++) {
        int s1 =
            (e >> 6  | e << 26) ^
            (e >> 11 | e << 21) ^
            (e >> 25 | e << 7);

        int ch =
            (e & f) ^ (~e & g);

        int t1 =
            h + s1 + ch + w[i];

        int s0 =
            (a >> 2  | a << 30) ^
            (a >> 13 | a << 19) ^
            (a >> 22 | a << 10);

        int maj =
            (a & b) ^ (a & c) ^ (b & c);

        int t2 =
            s0 + maj;

        // extra SHA-like local mixing, but no new long-lived state
        int m0 = s1 ^ s0;
        int m1 = ch + maj;
        int m2 = t1 ^ t2;
        int m3 = (m0 + m1) ^ m2;

        // use the mixes late, so they overlap with rotation
        h = g + (m0 & 7);
        g = f ^ (m1 & 3);
        f = e + (m2 & 1);
        e = d + t1 + (m3 & 15);
        d = c;
        c = b + (m0 & 1);
        b = a ^ (m1 & 1);
        a = t1 + t2 + (m3 & 31);
    }

    return a ^ b ^ c ^ d ^ e ^ f ^ g ^ h;
}


int storm3(
    int *w,
    int a,
    int b,
    int c,
    int d,
    int e,
    int f,
    int g,
    int h)
{
    for (int i = 0; i < 64; i++) {
        int s1 =
            (e >> 6  | e << 26) ^
            (e >> 11 | e << 21) ^
            (e >> 25 | e << 7);

        int ch =
            (e & f) ^ (~e & g);

        int t1 =
            h + s1 + ch + w[i];

        int s0 =
            (a >> 2  | a << 30) ^
            (a >> 13 | a << 19) ^
            (a >> 22 | a << 10);

        int maj =
            (a & b) ^ (a & c) ^ (b & c);

        int t2 =
            s0 + maj;

        int olda = a;
        int olde = e;

        int m0 = s1 ^ s0;
        int m1 = ch + maj;
        int m2 = t1 ^ t2;
        int m3 = olda + olde;
        int m4 = (m0 ^ m1) + (m2 ^ m3);

        // masked late uses: keep storm2's backend annoyance
        int k0 = m0 & 7;
        int k1 = m1 & 3;
        int k2 = m2 & 1;
        int k3 = m3 & 15;
        int k4 = m4 & 31;

        h = g + k0;
        g = f ^ k1;
        f = e + k2;
        e = d + t1 + k3;
        d = c + m3;
        c = b ^ k0;
        b = olda + t2;
        a = t1 + t2 + k4;
    }

    return a ^ b ^ c ^ d ^ e ^ f ^ g ^ h;
}