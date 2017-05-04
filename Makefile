all:
	 g++ main.cpp -o Rosto `pkg-config --cflags --libs opencv`
	 
clean:
	  rm Rosto
