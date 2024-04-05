#!/usr/bin/awk -f

BEGIN {
    RS=","
    ORS=","
    }


    {
	newline=""
    }

/[^0-9]/   {
       newline="\n";
       gsub("[ \t\r\n]","");
   }

/^$/ { next }



   {
       v=$0+0
       rv=0;
       mask1=128;
       mask2=1;
       for (mask1=128;mask1>=1;mask1=mask1/2) {
          if (v >= mask1)
	  {
	      rv=rv+mask2;
	      v=v-mask1;
	  }
	  mask2=mask2*2;
       }
       print newline rv;
   }

