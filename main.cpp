/*
 * main.cpp
 *
 *  Created on: 07/12/2016
 *      Author: jpeumesmo
 */



#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string>
#include <sstream>
#include <time.h>

namespace patch{
    template < typename T > std::string to_string( const T& n )
    {
        std::ostringstream stm ;
        stm << n ;
        return stm.str() ;
    }
}

using namespace cv;

String exec(const char* cmd) {
    char buffer[128];
    std::string result = "";
    FILE* pipe = popen(cmd, "r");
    if (!pipe) throw std::runtime_error("popen() failed!");
    try {
        while (!feof(pipe)) {
            if (fgets(buffer, 128, pipe) != NULL)
                result += buffer;
        }
    } catch (...) {
        pclose(pipe);
        throw;
    }
    pclose(pipe);
    return result;
}

void locate(String &buff){
  std::string in;
  char aux[256];
  in = exec("locate data/haarcascades/haarcascade_frontalface_alt.xml");
  buff = in;
}


void TSL (Mat& in, Mat& out){

	for (int i = 0; i < in.rows; i++){
		for (int j = 0; j < in.cols; j++){

			cv::Vec3f intensity = in.at<Vec3f>(i,j);
			float blue=intensity.val[0];
			float green=intensity.val[1];
			float red=intensity.val[2];

			//SOMATORIO
			double sum=red+blue+green;

			//NORMALIZADO
			double r= red/sum;
			double g= green/sum;

			//R' E G'
			double r_ = r - (1/3);
			double g_ = g - (1/3);


			//CALCULA O L
			double L = (0.299*red) + (green*0.587) + (blue*0.114);


			//CALCULA O S
			double S = sqrt((9/5)*(pow(r_,2)+pow(g_,2)));

			//CALCULA O T
			double T;
			if (g_ == 0) {
				T = 0;
			}
			if (g_ < 0) {
				T = ( (1/(2*3.14)) * atan((r_/g_) + (3/4) ) ) ;
			}
			if (g_ > 0) {
				T = ( (1/(2*3.14)) * atan((r_/g_) + (1/4) ) ) ;
			}
			/* MONTA A SAIDA

			A.data[A.step[0]*i + A.step[1]* j + 0] = (b*255);
				 A.data[A.step[0]*i + A.step[1]* j + 1] = (g*255);
				 A.data[A.step[0]*i + A.step[1]* j + 2] = (r*255);
				*/

			out.data[out.step[0]*i + out.step[1]*j + 0] = T;
			out.data[out.step[0]*i + out.step[1]*i + 1] = S;
			out.data[out.step[0]*i + out.step[1]*i + 2] = L;
		}
	}
  //imshow("teste",T);
	imshow("teste",out);
}

void balancear(Mat& in, Mat& out, float percent) {
	assert(in.channels() == 3);
	assert(percent > 0 && percent < 100);

	float half_percent = percent / 200.0f;

	std::vector<Mat> tmpsplit; split(in,tmpsplit);
	for(int i=0;i<3;i++) {
		//find the low and high precentile values (based on the input percentile)
		Mat flat; tmpsplit[i].reshape(1,1).copyTo(flat);
		cv::sort(flat,flat,CV_SORT_EVERY_ROW + CV_SORT_ASCENDING);
		int lowval = flat.at<uchar>(cvFloor(((float)flat.cols) * half_percent));
		int highval = flat.at<uchar>(cvCeil(((float)flat.cols) * (1.0 - half_percent)));
		// std::cout << lowval << " " << highval << "\n";

		//saturate below the low percentile and above the high percentile
		tmpsplit[i].setTo(lowval,tmpsplit[i] < lowval);
		tmpsplit[i].setTo(highval,tmpsplit[i] > highval);

		//scale the channel
		normalize(tmpsplit[i],tmpsplit[i],0,255,NORM_MINMAX);
	}
	merge(tmpsplit,out);
}

bool compareFacePosition ( const Rect & face1, const Rect & face2 ) {
	int x1 = face1.x;
	int x2 = face2.x;
	//double i = fabs( face1.area() );
	//double j = fabs( face2.area() );
	return ( x1 > x2 );
}

bool compareContourAreas ( std::vector<cv::Point> contour1, std::vector<cv::Point> contour2 ) {
	double i = fabs( contourArea(cv::Mat(contour1)) );
	double j = fabs( contourArea(cv::Mat(contour2)) );
	return ( i > j );
}

bool compareContourPosition(std::vector<cv::Point> contour1, std::vector<cv::Point> contour2 ) {
	int x1 = contour1.at(2).x;
	int x2 = contour2.at(2).x;
	return (x1 > x2);
}

bool compareContourConvexity ( std::vector<cv::Point> contour1, std::vector<cv::Point> contour2 ) {
	std::vector<std::vector<int> >hull (2);
	std::vector<std::vector<Vec4i> > defects(2);


	convexHull(contour1, hull[0], false);
	convexHull(contour2, hull[1], false);
	convexityDefects(contour1, hull[0], defects[0]);
	convexityDefects(contour2, hull[1], defects[1]);

	int i = defects[0].size();
	int j = defects[1].size();

	return ( i > j );
}

void binarizar(Mat &in, Mat& out){

	Mat element = getStructuringElement( 0,
			Size( 2*3+ 1, 2*3+1 ),
			Point( 3, 3) );Mat kernel = (Mat_<int>(3, 3) <<
        0, 1, 0,
        1, -1, 1,
        0, 1, 0);


	Mat blured, hsv, balanceada;



	blur( in, blured, Size(13,13) );
//  blur( in, blured, Size(7,7) );


	//balancear(blured,balanceada,10);

	//cvtColor(balanceada,hsv,CV_BGR2HSV);
	cvtColor(blured, hsv, CV_BGR2HSV);
//	balancear(hsv,balanceada,10);
	//cvtColor(in,hsv,CV_BGR2HSV);

//  inRange(balanceada, Scalar(0, 40, 60), Scalar(15, 150, 255), out);

	inRange(hsv, Scalar(0, 40, 60), Scalar(15, 150, 255), out);
//  inRange(hsv, Scalar(0, 48, 80), Scalar(20, 255, 255), out);

  morphologyEx(out,out,1,element);
  //morphologyEx( out, out, MORPH_CLOSE, element );

	//std::vector<Mat> channels;
	//split(hsv,channels);
	//imshow("H",channels[0]);
	//imshow("S",channels[1]);
	//imshow("V",channels[2]);
}

//void tracking(Ptr<Tracker> trackerFace,Rect2d faceBox,Ptr<Tracker> trackerLeft,Rect2d leftBox,Ptr<Tracker> trackerRigth,Rect2d rigthBox ){

  //bbox = boundingRect( Mat(contours[0]) );
  //tracker->update(frame, bbox);
  //rectangle(frame, bbox, azul, 2, 1 );

//}

int main( int argc, char** argv ){

  time_t rawtime;
struct tm * timeinfo;
int flagTrack = 0;
int exPtIniTrack,radiusIniTrack, eyPtIniTrack,contTrack =0;
int dxPtIniTrack, dyPtIniTrack;
int cont;

Mat kernel = (Mat_<int>(3, 3) <<
        0, 1, 0,
        1, -1, 1,
        0, 1, 0);


  //VARIAVEIS
	Scalar verde = Scalar(0,255,0);
	Scalar azul = Scalar(255,0,0);
	Scalar vermelho = Scalar(0,0,255);
	CascadeClassifier face_cascade;
    Mat frame,gray,bin,aux,hsv,blured;
	int cntr = 0;
    unsigned int numeroPessoas;
    String aux_string, localCascade;


//  Ptr<Tracker> trackerLeft = TrackerTLD::create();
//  Ptr<Tracker> trackerRigth = TrackerTLD::create();

  Ptr<Tracker> trackerLeft = TrackerKCF::create();
  Ptr<Tracker> trackerRigth = TrackerKCF::create();

//  Rect2d leftBox,rigthBox;

//  trackerLeft->init(frame,leftBox);
//  trackerRigth->init(frame,rigthBox);

  //ACHA  PATH DO CASCADE
  locate(aux_string);
  localCascade = aux_string.substr (0, aux_string.length()-1 );

	//INCIALIZA O DISPOSITIVO
	VideoCapture cap(0);


	//VERIFICA SE O DISPOSITIVO FOI INICIALIZADO CORRETAMENTE
	if(!cap.open(0)){
		return 0;
	}

	//LOOP PRINCIPAL
	while(1){
		//capturar frame
    cap >> frame;


    flip(frame, frame, 1);

		cvtColor(frame,gray,CV_BGR2GRAY);
    equalizeHist(gray, gray );


    /*if( !face_cascade.load( "/home/jpeumesmo/Applications/OpenCV/opencv-3.1.0/data/haarcascades/haarcascade_frontalface_alt.xml") ){
      printf("--(!)Error loading\n");
      return -1;
    }*/
    if( !face_cascade.load("haarcascade_frontalface_alt.xml") ){
      printf("--(!)Error loading\n");
      return -1;
    }
		//face_cascade.load( "haarcascade_frontalface_alt.xml" ) ;

		std::vector<Rect> faces;

		face_cascade.detectMultiScale( gray, faces, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );

		numeroPessoas = faces.size();

    aux = frame.clone();
		binarizar(aux,bin);
		blur( bin, blured, Size(7,7) );
		imshow("Binaria",blured);

		std::vector<std::vector<Point> > contours;
		std::vector<Vec4i> hierarchy;
    std::vector<Moments> mu(2);
    std::vector<Point2f> mc(2);
    std::vector<std::vector<Point> > hull(2);
    std::vector<std::vector<int> > hulli(2);
    std::vector<std::vector<Vec4i> > defects(2);


		findContours( bin, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

		std::sort(contours.begin(), contours.end(), compareContourAreas);


		switch (numeroPessoas){


		case(0):{
			/*
			 * Caso de nenhuma pessoa detectada
			 */
			putText(frame,"Nenhuma pessoa detectada",Point(200,50),FONT_HERSHEY_SIMPLEX,1,vermelho,1,8);
			break;



		}
		case(1):{
			/*
			 * Caso de uma pessoa detectada
			 */

			sort(contours.begin(),contours.begin()+3,compareContourPosition);

			Point centerUnico( faces[0].x + faces[0].width*0.5, faces[0].y + faces[0].height*0.5 );
			ellipse( frame, centerUnico, Size( faces[0].width*0.5, faces[0].height*0.5), 0, 0, 360, verde, 4, 8, 0 );
      if (!flagTrack){
        exPtIniTrack = faces[0].x - faces[0].width*0.5;
        eyPtIniTrack = frame.size().height*0.2;
        dxPtIniTrack = faces[0].x + faces[0].width*0.5;
        dyPtIniTrack = frame.size().height*0.2;
        radiusIniTrack = faces[0].width*0.25;
        //Point diniciarTrack(faces[0].x+faces[0].width*0.2,frame.size().height*0.5 );
        //Point einiciarTrack(faces[0].x+faces[0].width*0.2,frame.size().height*0.5 );
        flagTrack = 1;
      }

      /*
       PEGAR CENTRO DE MASSA
       MAO DIREITA COR AZUL INDICE 0
       MAO ESQUERDA COR VERMELHA INDICE 2
      */

      mu[0] = moments( contours[0], false );
      mu[1] = moments( contours[2], false );
      mc[0] = Point2f( mu[0].m10/mu[0].m00 , mu[0].m01/mu[0].m00 );
      mc[1] = Point2f( mu[1].m10/mu[1].m00 , mu[1].m01/mu[1].m00 );

    convexHull( Mat(contours[0]), hull[0], false );
    convexHull( Mat(contours[2]), hull[1], false );

    convexHull( Mat(contours[0]), hulli[0], false );
    convexHull( Mat(contours[2]), hulli[1], false );


    convexityDefects( Mat(contours[0]), hulli[0], defects[0] );
    convexityDefects( Mat(contours[2]), hulli[1], defects[1] );

//      morphologyEx(Mat(contours[0]), Mat(hull[0]), MORPH_HITMISS, kernel);
//      morphologyEx(Mat(contours[2]), Mat(hull[1]), MORPH_HITMISS, kernel);

//    Point diniciarTrack(faces[0].x - faces[0].width*0.5,dyPtIniTrack);
//    circle(frame,diniciarTrack,radiusIniTrack,vermelho,1,8,0);
//    Point einiciarTrack(faces[0].x + faces[0].width*1.5,eyPtIniTrack);
//    circle(frame,einiciarTrack,radiusIniTrack,azul,1,8,0);

//------------------------------------------------------------------------------
/*

      drawContours(frame,contours,0,verde,-1,8,hierarchy);

      circle( frame, mc[0], 8, azul, -1, 8, 0 );
      drawContours( frame, hull, 1, azul, 1, 8, std::vector<Vec4i>(), 0, Point() );
*/
// std::cout <<hull[1].size() << "\t"<< defects[1][1].size() << '\n';
//for (int k = 0; k < hull[1].size(); k++){
          cont = 0;
            for (int j = 0; j < defects[1].size(); j++){
              if (defects[1][j][3] > 20 * 256 /*filter defects by depth*/){
                int ind_0 = defects[1][j][0];//start point
                int ind_1 = defects[1][j][1];//end point
                int ind_2 = defects[1][j][2];//defect point
                if (contours[2][ind_0].y < mc[1].y){
                    cont++;
                  circle(frame, contours[2][ind_0], 5, Scalar(0, 0, 255), -1);
            //circle(frame, contours[2][ind_1], 5, Scalar(255, 0, 0), -1);
                //    circle(frame, contours[2][ind_2], 5, Scalar(0, 255, 0), -1);
                //    line(frame, contours[2][ind_2], contours[2][ind_0], Scalar(0, 255, 255), 1);
                //    line(frame, contours[2][ind_2], contours[2][ind_1], Scalar(0, 255, 255), 1);

                }
              }
            }
            switch (cont) {
              case 5:{
                putText(frame,"5 dedos",Point(200,50),FONT_HERSHEY_SIMPLEX,1,vermelho,1,8);
              }

              break;

              case 4:{
                putText(frame,"4 dedos",Point(200,50),FONT_HERSHEY_SIMPLEX,1,vermelho,1,8);
              }

              break;

              case 3:{
                putText(frame,"3 dedos",Point(200,50),FONT_HERSHEY_SIMPLEX,1,vermelho,1,8);
              }

              break;

              case 2:{
                putText(frame,"2 dedos",Point(200,50),FONT_HERSHEY_SIMPLEX,1,vermelho,1,8);
              }

              break;

              case 1:{
                putText(frame,"1 dedo",Point(200,50),FONT_HERSHEY_SIMPLEX,1,vermelho,1,8);
              }

              break;

              case 0:{
                putText(frame,"Nenhum dedo",Point(200,50),FONT_HERSHEY_SIMPLEX,1,vermelho,1,8);
              }

              break;

              default:{
                break;
              }
            }
  //        }
/*
      for (const Vec4i& v : defects[1]){
        int startidx = v[0]; Point ptStart(contours[0][startidx]);
        int endidx = v[1]; Point ptEnd(contours[0][endidx]);
        int faridx = v[2]; Point ptFar(contours[0][faridx]);

        line(frame,ptStart,ptEnd,vermelho,1);
        line(frame,ptStart,ptFar,vermelho,1);
        line(frame,ptEnd,ptFar,vermelho,1);
        circle(frame,ptFar,4,vermelho,2);
      }
*/
      //drawContours(frame,contours,2,verde,-1,8,hierarchy);
      circle( frame, mc[1], 8, vermelho, -1, 8, 0 );
      //if ( ((mc[1].x - dxiniciarTrack.x) * (mc[1].x - diniciarTrack.x)) + ((mc[1].y - diniciarTrack.y) * (mc[1].y - diniciarTrack.y)) < (radiusIniTrack * radiusIniTrack )
      //    &&  ((mc[0].x - diniciarTrack.x) * (mc[0].x - diniciarTrack.x)) + ((mc[0].y - diniciarTrack.y) * (mc[0].y - diniciarTrack.y)) < (radiusIniTrack * radiusIniTrack )

//    ){
//        std::cout << contTrack << '\n';
  //      contTrack++;
  //      if (contTrack == 3){
    //      std::cout << "tracking" << '\n';
    //    }
    //  }
    //  drawContours( frame, hull, 1, azul, 1, 8, std::vector<Vec4i>(), 0, Point() );

//      time ( &rawtime );
//  timeinfo = localtime ( &rawtime );
//  fprintf ( "%d    Data atual do sistema é: %s ",rawtime , asctime (timeinfo));

			break;
		}

		default:{
			/*
			 * Mais do que 3 pessoas detectadas
			 */
			putText(frame,"Pessoas demais detectadas",Point(200,50),FONT_HERSHEY_SIMPLEX,1,vermelho,1,8);

			break;
		}
		}

		imshow("frame",frame);

		cntr++;
		imwrite("/home/jpeumesmo/workspace/Rosto/images/bin/"+patch::to_string(cntr)+".jpg",bin);
		imwrite("/home/jpeumesmo/workspace/IC/Rosto/images/frame/f"+patch::to_string(cntr)+".jpg",frame);

		if(waitKey(30) >= 0) break;//QUEBRA LACO PRINCIPAL
	}

	return 0;
}
