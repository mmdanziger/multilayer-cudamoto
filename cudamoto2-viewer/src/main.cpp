#include "cudamotoviewer.h"
#include <QApplication>

int main(int argc, char *argv[])
{
    srand(time(0));
    QApplication a(argc, argv);
    QApplication::setAttribute(Qt::AA_EnableHighDpiScaling);
    CudamotoViewer w;
    w.show();

    return a.exec();
}
