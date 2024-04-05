import java.lang.*;   // support for System.out

public class Fun {    // must have same name as file
	public static void main(String[] args) {  // main function
	  double y[] = new double[11];     // array of doubles
    
      for (int i=0;i<11;i++) {
        double x = -1.0 + 2.0*i/10;
        y[i] = fun(x);
      }

      // similar to C printf
	  System.out.printf("f(-1)=%g, f(1)=%g\n",y[0],y[10]);
	}

    // static since not associated with any object instance
	static double fun(double x) {
      double z =x*x - 2*x;
	  if (z>=0) return z; else return 0;
   }
}

