#include "NN_GPC.h"
#include "Wire.h"
#include <Servo.h>
#include <math.h>

#define PINIONSERVO_PIN 3
#define TWISTSERVO_PIN 44
#define BENDSERVO_PIN 2

float pi=3.14159;
float twist=0;//twist angle
int spot=0;//position of ring
int times = 0;//number of cycles of angles
int count = 0;//number of angles
int wait = 2000;//time to wait b4 data taken in ms
int maxstep=18;//number of angles
int n=0;//number of times position of ring moves
int off = 90;//ring servo off value
int done = 0;
float window[14];

void setup() {
  Serial.begin(9600);
  
  pinionServo.attach(PINIONSERVO_PIN);
  pinionServo.write(off);
 
  twistServo.attach(TWISTSERVO_PIN);
  twistServo.write(0);

  bendServo.attach(BENDSERVO_PIN);
  bendServo.write(90);
  
  delay(5000);
  twist=(-pi/2+float(twistServo.read())/180*pi)*.67778;//twist angle
  
  buildLayers(); // SARAH: added this to build the neural network layers. 

}

void loop() {
  if (done==0){
    if (count<=maxstep){//cycle of twist angles
      twistServo.write(180/maxstep*count);
      delay(wait);
      pinionServo.write(off);
      count++;
      twist=(-pi/2+float(twistServo.read())/180*pi)*.67778;
      
      takeData();
    }
    else if (times<10){//repeat cycle of twist angles
      twistServo.write(0);
      pinionServo.write(off);
      delay(wait);
      count=1;
      twist=(-pi/2+float(twistServo.read())/180*pi)*.67778;
      //
      takeData();
      u_optimal = NN_GPC(window); // SARAH: added this line to figure out best output
      //
      times++;
    }
    else {//move position of ring
      if (n<=10){
         pinionServo.write(0);
         delay(100);
         pinionServo.write(off);
         delay(500);
         twistServo.write(0);
         delay(wait);
         n++;
         spot = n*8.6;
         times=0;
         count=1;
         twist=(-pi/2+float(twistServo.read())/180*pi)*.67778;
         //
         takeData(); 
         u_optimal = NN_GPC(window); // SARAH: added this line to figure out best output
         //
       }
       else{
          done = 1;
       }
    }
  }
  else{//restet to neutral position
    twistServo.write(90);
    if (n==11){
      pinonServo.write(180);
      delay(1000);
      pinionServo.write(off);
      n++;
    }
  }
}

void takeData(){

      window[0] = analogRead(A0);
      window[1] = analogRead(A1);
      window[2] = analogRead(A2);
      window[3] = analogRead(A3);
      window[4] = analogRead(A4);
      window[5] = analogRead(A5);
      window[6] = analogRead(A6);
      window[7] = analogRead(A7);
      window[8] = analogRead(A8);
      window[9] = spot;
      window[10] = sin(twist);
      window[11]= cos(twist);
  }
