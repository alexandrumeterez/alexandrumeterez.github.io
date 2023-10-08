.PHONY : clean
clean :
	rm *.html	

.PHONE : website
website : 
	./jemdoc -c mysite.conf *.jemdoc