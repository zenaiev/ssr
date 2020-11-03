// https://stackoverflow.com/questions/24988164/c-fast-screenshots-in-linux-for-use-with-opencv

// g++ test-ximage.cxx -o test-ximage -lX11 -lXext `$cv`-Ofast -mfpmath=both -march=native -m64 -funroll-loops -mavx2 -lopencv_core -lopencv_highgui && ./test-ximage

#include <opencv2/stitching.hpp>

#include <X11/Xlib.h>
#include <X11/Xutil.h>

#include <X11/extensions/XShm.h>
#include <sys/ipc.h>
#include <sys/shm.h>

#include <opencv2/opencv.hpp>  // This includes most headers!
//#include <boost/compute/interop/opencv.hpp>  // This includes most headers!

#include <time.h>
#define FPS(start) (CLOCKS_PER_SEC / (clock()-start))

// Using one monitor DOESN'T improve performance! Querying a smaller subset of the screen DOES
const uint WIDTH  = 1920>>0;
const uint HEIGHT = 1080>>0;

// -------------------------------------------------------
int main(){
    Display* display = XOpenDisplay(NULL);
    Window root = DefaultRootWindow(display);  // Macro to return the root window! It's a simple uint32
    XWindowAttributes window_attributes;
    XGetWindowAttributes(display, root, &window_attributes);
    Screen* screen = window_attributes.screen;
    XShmSegmentInfo shminfo;
    XImage* ximg = XShmCreateImage(display, DefaultVisualOfScreen(screen), DefaultDepthOfScreen(screen), ZPixmap, NULL, &shminfo, WIDTH, HEIGHT);

    shminfo.shmid = shmget(IPC_PRIVATE, ximg->bytes_per_line * ximg->height, IPC_CREAT|0777);
    shminfo.shmaddr = ximg->data = (char*)shmat(shminfo.shmid, 0, 0);
    shminfo.readOnly = false;
    if(shminfo.shmid < 0)
        puts("Fatal shminfo error!");;
    Status s1 = XShmAttach(display, &shminfo);
    printf("XShmAttach() %s\n", s1 ? "success!" : "failure!");

    cv::Mat img;

    for(int i; ; i++){
        double start = clock();

        XShmGetImage(display, root, ximg, 0, 0, 0x00ffffff);
        img = cv::Mat(HEIGHT, WIDTH, CV_8UC4, ximg->data);

        if(!(i & 0b111111))
            printf("fps %4.f  spf %.4f\n", FPS(start), 1 / FPS(start));
        break;
    }

    cv::imshow("img", img);
    cv::waitKey(0);

    XShmDetach(display, &shminfo);
    XDestroyImage(ximg);
    shmdt(shminfo.shmaddr);
    XCloseDisplay(display);
    puts("Exit success!");
}
