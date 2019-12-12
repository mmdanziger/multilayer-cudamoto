#include "cudamotoviewer.h"
#include "ui_cudamotoviewer.h"


CudamotoViewer::CudamotoViewer(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::CudamotoViewer),N(1<<10),k1(6),k2(6),f(1),h(0.01),refresh_interval(100)
{
    ui->setupUi(this);
    populate_color_table();
    running=false;
    fromData = 0;
    initialize();
}

CudamotoViewer::~CudamotoViewer()
{
    cm2.reset();
    delete ui;
}


void CudamotoViewer::initialize(){
    grab_parameters_from_widget();
    cm2.reset();
    if(fromData){
        cm2 = std::unique_ptr<Cudamoto2>(new Cudamoto2(edgeListFname.toStdString()));
    } else {
        cm2 = std::unique_ptr<Cudamoto2>(new Cudamoto2(N,k1,k2,f,h));
    }
    if(isMixed){
        cm2->make_mixed();
    }
    L = static_cast<uint>(ceil(sqrt(N)));
    lattice1 = QImage(L,L,QImage::Format_Indexed8);
    lattice2 = QImage(L,L,QImage::Format_Indexed8);
    lattice1.setColorTable(color_table);
    lattice2.setColorTable(color_table);
    cm2->set_interaction(interaction_type);
    cm2->set_cauchy_tail_parameter(natural_distribution_parameter);
    cm2->set_distribution(natural_distribution_type);
    cm2->generate_omega();
    cm2->copy_omega_to_device();
    burstState = false;
    ui->burstButton->setText("Burst On");
    if(topology_type ==1){
        cm2->set_zeta(zeta1,zeta2);
        cm2->generate_exponential_length_networks();
    }
    if(isDirected)
        cm2->make_directed();

    drawLattices();

}

void CudamotoViewer::populate_color_table(){
    color_table.clear();
    int s=255,v=255;
    for(int i=0; i<360; ++i){
        color_table.push_back( QColor::fromHsv(i,s,v).rgb());
    }

}

void CudamotoViewer::drawLattices(){

    cm2->copy_theta_to_host();
    const std::vector<float> theta = cm2->get_theta();
    int color_index;
    for(uint i=0; i<N; ++i){
        color_index = static_cast<int>(round(theta[i])*ONE_EIGHTY_OVER_PI_DEF);
        do{
        color_index = (color_index + 360)%360;
        }while(color_index<0);
        lattice1.setPixel(i%L,i/L, color_index);
    }
    for(uint i=0; i<N; ++i){
        color_index = static_cast<int>(round(theta[i+N]*ONE_EIGHTY_OVER_PI_DEF));
        do{
        color_index = (color_index + 360)%360;
        }while(color_index<0);
        lattice2.setPixel(i%L,i/L, color_index);
    }

    mscene1.clear(); mscene2.clear();

    mscene1.addPixmap(QPixmap::fromImage(lattice1));
    mscene1.setSceneRect(lattice1.rect());
    this->ui->layerView1->setScene(&mscene1);
    this->ui->layerView1->setRenderHints(QPainter::Antialiasing | QPainter::SmoothPixmapTransform);

    mscene2.addPixmap(QPixmap::fromImage(lattice2));
    mscene2.setSceneRect(lattice2.rect());
    this->ui->layerView2->setScene(&mscene2);
    this->ui->layerView2->setRenderHints(QPainter::Antialiasing | QPainter::SmoothPixmapTransform);

    this->ui->layerView1->show();
    this->ui->layerView2->show();

    float2 r = cm2->get_r_device();
    this->ui->r1label->setText(QString::number(r.x));
    this->ui->r2label->setText(QString::number(r.y));
    this->ui->tlabel->setText(QString::number(t));
    if(burstState && t > burstInitialTime + burstIndex * burstImageEvery){
        save_burst_images();
    }


}

void CudamotoViewer::do_local_attack(int net_idx, int target_state){

    int rh,restore=0;
    if(running)
    {
  //      running = false;
        restore = 1;
    }
    if (net_idx == 0)
        rh = this->ui->r1h_edit->text().toInt();
    else
        rh = this->ui->r2h_edit->text().toInt();
    cm2->local_attack(rh,net_idx,target_state);
    drawLattices();
    //if(restore)
      //  run();
}

void CudamotoViewer::on_zoomSlider_valueChanged(int value)
{
    qreal scale = qPow(qreal(2), (value - 250) / qreal(40));

    QMatrix matrix;
    matrix.scale(scale, scale);
    this->ui->layerView1->setMatrix(matrix);
    this->ui->layerView2->setMatrix(matrix);
}

void CudamotoViewer::on_startButton_clicked()
{
    running = !running;
    QString newText = running? "Stop" : "Start";
    this->ui->startButton->setText(newText);
    if(running){
        initialize();
        run();
    }
}

void CudamotoViewer::run(){
    uint step_counter=0;
    t=0;
    cm2->set_lambda(lambda1,lambda2);
    while(running){
        cm2->integrate_one_step();
        step_counter++;
        //std::cout << step_counter << std::endl;
        QCoreApplication::processEvents();
        t+=h;
        if(step_counter%refresh_interval == 0){
           drawLattices();
//            running=false;
        }

    }

}

void CudamotoViewer::stop()
{
    running = false;
    QString newText = "Start";
    this->ui->startButton->setText(newText);
}

void CudamotoViewer::grab_parameters_from_widget(){
    N =  this->ui->sizeBox->currentText().toInt();
    k1 = this->ui->k1Edit->text().toFloat();
    k2 = this->ui->k2Edit->text().toFloat();
    zeta1 = this->ui->zeta1Edit->text().toFloat();
    zeta2 = this->ui->zeta2Edit->text().toFloat();
    f = this->ui->fEdit->text().toFloat();
    lambda1 = this->ui->lambda1Edit->text().toFloat();
    lambda2 = this->ui->lambda2Edit->text().toFloat();
    interaction_type = this->ui->interactionBox->currentIndex()+1;
    topology_type = this->ui->topologyBox->currentIndex();
    natural_distribution_type = this->ui->naturalFrequencyBox->currentIndex()+1;
    natural_distribution_parameter = this->ui->naturalFrequencyParameter->text().toFloat();
    isDirected = this->ui->isDirectedCheckBox->isChecked() ? 1 : 0;
    isMixed = this->ui->isMixedCheckBox->isChecked() ? 1 : 0;
}

void CudamotoViewer::on_sizeBox_currentIndexChanged(int index)
{
    stop();
    grab_parameters_from_widget();
}

void CudamotoViewer::on_k1Edit_textChanged(const QString &arg1)
{
    stop();
    grab_parameters_from_widget();
}


void CudamotoViewer::on_k2Edit_textChanged(const QString &arg1)
{
    stop();
    grab_parameters_from_widget();
}

void CudamotoViewer::on_lambda1Edit_textEdited(const QString &arg1)
{

    grab_parameters_from_widget();
    cm2->set_lambda(lambda1,lambda2);
}


void CudamotoViewer::on_lambda2Edit_textEdited(const QString &arg1)
{

    grab_parameters_from_widget();
    cm2->set_lambda(lambda1,lambda2);
}

void CudamotoViewer::on_fEdit_textChanged(const QString &arg1)
{


    grab_parameters_from_widget();
    cm2->set_f(f);
}


void CudamotoViewer::on_interactionBox_currentIndexChanged(int index)
{
    grab_parameters_from_widget();
    cm2->set_interaction(interaction_type);

}

void CudamotoViewer::on_lamSumSlider_valueChanged(int value)
{
    //std::cout << "Changing lambdas from ("<<lambda1<<","<<lambda2<<") to (";
    lambda_sum = lambda1 + lambda2;
    lambda1 = lambda_sum *(1 - ( static_cast<float>(this->ui->lamSumSlider->value()) / static_cast<float>(this->ui->lamSumSlider->maximum())));
    lambda2 = lambda_sum - lambda1;
    //std::cout <<lambda1<<","<<lambda2<<")\n";
    cm2->set_lambda(lambda1,lambda2);
    //cm2->make_competitive();

    this->ui->lambda2Edit->setText(QString::number(lambda2));
    this->ui->lambda1Edit->setText(QString::number(lambda1));
}

void CudamotoViewer::on_r1SyncButton_clicked()
{
    cm2->zero_oscillator_phases(0);
}

void CudamotoViewer::on_r1RandButton_clicked()
{
    cm2->randomize_oscillator_phases(0);
}

void CudamotoViewer::on_r2SyncButton_clicked()
{
    cm2->zero_oscillator_phases(1);
}

void CudamotoViewer::on_r2RandButton_clicked()
{
    cm2->randomize_oscillator_phases(1);
}

void CudamotoViewer::on_r1hSyncButton_clicked()
{
    do_local_attack(0,1);
}

void CudamotoViewer::on_r1hRandButton_clicked()
{
    do_local_attack(0,0);
}

void CudamotoViewer::on_r2hSyncButton_clicked()
{
    do_local_attack(1,1);
}

void CudamotoViewer::on_r2hRandButton_clicked()
{
    do_local_attack(1,0);
}

void CudamotoViewer::on_topologyBox_currentIndexChanged(int index)
{
    stop();
    grab_parameters_from_widget();
}

void CudamotoViewer::on_zeta1Edit_textChanged(const QString &arg1)
{
    stop();
    grab_parameters_from_widget();
}



void CudamotoViewer::on_zeta2Edit_textChanged(const QString &arg1)
{
    stop();
    grab_parameters_from_widget();

}



void CudamotoViewer::on_saveImageButton_clicked()
{
    QString defaultfname = "/tmp/Cudamoto" +ui->interactionBox->currentText() + "_f" + QString::number(f) +
            "_kOne" + QString::number(k1) + "_kTwo" + QString::number(k1) +
            "_lamOne" + QString::number(lambda1) + "_lamTwo" + QString::number(lambda2) +
            "_rOne" + this->ui->r1label->text() + "_rTwo" + this->ui->r2label->text() + ".png";
    auto basefname = QFileDialog::getSaveFileName(this, tr("Save pair of images"), defaultfname, tr("Images (*.png *.jpg)"));
    if(basefname.length() > 0){
        QString extension = basefname.section(".",-1);
        QString filename = basefname.section(".",0,-2);
        QString fname1 = filename + "_net1." + extension;
        QString fname2 = filename + "_net2." + extension;
        std::cout << "Saving image to  " << fname1.toStdString() << std::endl;
        lattice1.save(fname1,extension.toStdString().c_str(), 100);
        lattice2.save(fname2,extension.toStdString().c_str(), 100);
    }
}

void CudamotoViewer::on_naturalFrequencyBox_currentIndexChanged(int index)
{
    grab_parameters_from_widget();
    cm2->set_cauchy_tail_parameter(natural_distribution_parameter);
    cm2->generate_omega();
    cm2->copy_omega_to_device();
    if(index == 1)
        this->ui->naturalFrequencyParameter->setReadOnly(false);
    else if(index ==0)
        this->ui->naturalFrequencyParameter->setReadOnly(true);


}

void CudamotoViewer::on_isDirectedCheckBox_stateChanged(int arg1)
{
    if (this->ui->isDirectedCheckBox->isChecked()){
        cm2->make_directed();
        isDirected = 1;
    }
}

void CudamotoViewer::on_burstButton_clicked()
{
    burstState = !burstState;
    if(burstState){
        burstImageEvery = ui->burstEvery->text().toFloat();
        burstIndex=0;
        burstInitialTime=t;
        burstFname = "/tmp/Cudamoto" +ui->interactionBox->currentText() + "_f" + QString::number(f) +
                "_kOne" + QString::number(k1) + "_kTwo" + QString::number(k1) +
                "_lamOne" + QString::number(lambda1) + "_lamTwo" + QString::number(lambda2) +
                "_rOne" + this->ui->r1label->text() + "_rTwo" + this->ui->r2label->text();
        ui->burstButton->setText("Burst Off");
    } else {
        ui->burstButton->setText("Burst On");


    }
}

void CudamotoViewer::save_burst_images(){
    lattice1.save(burstFname +"_t" + QString::number(this->t) + "_net1.png","png", 100);
    lattice2.save(burstFname +"_t" + QString::number(this->t) + "_net2.png","png", 100);
    burstIndex++;
}

void CudamotoViewer::on_actionLoad_triggered()
{
    stop();
    edgeListFname = QFileDialog::getOpenFileName(this, tr("Load edge list"),
                                                    "/home");
    fromData = 1;
    initialize();
}
