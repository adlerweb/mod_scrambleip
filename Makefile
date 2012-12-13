# Makefile for mod_scrambleip.c (gmake)
# $Id:$

#APXS2=/path/to/your/apxs
APXS2=`which apxs2`

default:
	@echo mod_scrambleip:
	@echo author: git@adlerweb.info
	@echo
	@echo following options available:
	@echo \"make scrambleip\" to compile the 2.0 version
	@echo \"make install\" to install the 2.0 version
	@echo
	@echo change path to apxs if this is not it: \"$(APXS2)\"


scrambleip: mod_scrambleip.o
	@echo make done, type \"make install\" to install mod_scrambleip-2.0

mod_scrambleip.o: mod_scrambleip.c
	$(APXS2) -c -n $@ mod_scrambleip.c MbotCexplode.c

mod_scrambleip.c:

install: mod_scrambleip.o
	$(APXS2) -i -n mod_scrambleip.so mod_scrambleip.la

clean:
	rm -rf *~ *.o *.so *.lo *.la *.slo .libs/
