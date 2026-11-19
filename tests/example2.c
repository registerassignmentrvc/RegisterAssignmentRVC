extern int x, y;
extern int y1, y2, y3, y4;

extern int bar(int arg0, int arg1, int arg2, int arg3);

static void
foo (int arg0, int arg1, int arg2, int arg3) __attribute__ ((noinline))
{
 
                                                                 
  x  = bar (arg0, arg1, arg2, arg3);
                                            
  y = (arg1 - arg2) + (arg3 - arg2 >> 2);

}

int main()
{
    foo(y1, y2, y3, y4);
}