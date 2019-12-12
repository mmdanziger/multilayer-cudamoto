#ifndef CUDAMOTOVIEWER_H
#define CUDAMOTOVIEWER_H

#include <QMainWindow>
#include <QGraphicsScene>
#include <qmath.h>
#include <memory>
#include <QFileDialog>
#include "../../cudamoto2/src/Cudamoto2.h"

#define ONE_EIGHTY_OVER_PI_DEF 57.29577951308232
#define TWO_PI_DEF 6.283185307179586
namespace Ui {
class CudamotoViewer;
}

class CudamotoViewer : public QMainWindow
{
    Q_OBJECT

public:
    explicit CudamotoViewer(QWidget *parent = 0);
    ~CudamotoViewer();

private:
    Ui::CudamotoViewer *ui;
    uint N,L;
    float k1,k2;
    float zeta1,zeta2;
    float lambda1,lambda2,lambda_sum;
    float f,h,t;
    float natural_distribution_parameter;
    bool running;
    int refresh_interval,interaction_type,topology_type,natural_distribution_type,isDirected,fromData,isMixed;
    QVector<QRgb> color_table;
    QImage lattice1,lattice2;
    QGraphicsScene mscene1,mscene2;
    std::unique_ptr<Cudamoto2> cm2;
// Variables to track burst pictures
    bool burstState;
    float burstImageEvery;
    int burstIndex;
    float burstInitialTime;
    QString burstFname,edgeListFname;


protected:
    void initialize();
    void populate_color_table();
    void drawLattices();
    void grab_parameters_from_widget();
    void run();
    void stop();
    void pause();
    void do_local_attack(int net_idx, int target_state);
    void save_burst_images();
private slots:
    void on_zoomSlider_valueChanged(int value);
    void on_startButton_clicked();
    void on_sizeBox_currentIndexChanged(int index);
    void on_k1Edit_textChanged(const QString &arg1);
    void on_k2Edit_textChanged(const QString &arg1);
    void on_lambda1Edit_textEdited(const QString &arg1);
    void on_lambda2Edit_textEdited(const QString &arg1);
    void on_fEdit_textChanged(const QString &arg1);

    void on_interactionBox_currentIndexChanged(int index);
    void on_lamSumSlider_valueChanged(int value);
    void on_r1SyncButton_clicked();
    void on_r1RandButton_clicked();
    void on_r2SyncButton_clicked();
    void on_r2RandButton_clicked();
    void on_r1hSyncButton_clicked();
    void on_r1hRandButton_clicked();
    void on_r2hSyncButton_clicked();
    void on_r2hRandButton_clicked();
    void on_topologyBox_currentIndexChanged(int index);
    void on_zeta1Edit_textChanged(const QString &arg1);
    void on_zeta2Edit_textChanged(const QString &arg1);
    void on_saveImageButton_clicked();
    void on_naturalFrequencyBox_currentIndexChanged(int index);
    void on_isDirectedCheckBox_stateChanged(int arg1);
    void on_burstButton_clicked();
    void on_actionLoad_triggered();
};

#endif // CUDAMOTOVIEWER_H
