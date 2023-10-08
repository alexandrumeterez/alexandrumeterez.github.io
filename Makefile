.PHONY : clean
clean :
	rm *.html	

.PHONE : website
website : 
	python jem/jemdoc.py index.jemdoc