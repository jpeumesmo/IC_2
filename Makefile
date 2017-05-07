all:
	 g++ main.cpp -o teste `pkg-config --cflags --libs opencv`
	 
clean:
	  rm teste
