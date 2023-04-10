#include "mainwindow.h"
#include "ui_mainwindow.h"

#include "neuralnetwork.h"
#include "about.h"
#include <fstream>
#include <QString>
#include <string>
#include <QFileDialog>
#include <QElapsedTimer>
#include <QImage>
#include <QColor>

bool nnEnabled;

long double get_gray(QColor cl){
    return (255.0 - (cl.red() * 0.2989 + cl.green() * 0.587 + cl.blue() * 0.114))/255.0 * 0.99 + 0.01;
}

neuralnetwork *nt;

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    setWindowIcon(QIcon("D:\\NeuralNetwork.png"));
    nt = new neuralnetwork(784, 200, 10, 0.1);
    ui->setupUi(this);
    ui->NeuralNetworkTest->setEnabled(false);
    ui->OpenImage->setEnabled(false);
    ui->UserImageCheck->setEnabled(false);
    ui->UserImagePath->setEnabled(false);
    nnEnabled = false;

    QFile f( "D:\\NeuralNetwork\\style.stylesheet" );
    if ( !f.exists() )
    {
    //    qWarning() << "Unable to set dark stylesheet, file not found";
    }
    else
    {
        f.open( QFile::ReadOnly | QFile::Text );
        QTextStream ts( &f );

        this->setStyleSheet(ts.readAll());
    }

}

MainWindow::~MainWindow()
{
    delete nt;
    delete ui;
}


void MainWindow::on_actionOpen_triggered()
{
    QString path = QFileDialog::getOpenFileName(this,
        tr("Open Neural Network"), "D:\\", tr("Neural Network (*.ntdb)"));



        std::ifstream file;
        file.open(path.toStdString());

        std::vector<std::vector<long double>> wih;
        std::vector<std::vector<long double>> who;
        std::vector<long double> temp;
        std::string str = "";
        char ch;
        bool kl = true;
        while (file.good()) {
            ch = file.get();

            if(ch == '\\') {kl = false; str = ""; temp.clear(); continue;}

            if(kl){
                if (ch == '/'){
                        temp.push_back(std::stod(str));
                        str = "";
                    } else if (ch == '\n'){
                        wih.push_back(temp);
                        temp.clear();
                    } else {
                        str += ch;
                }
            } else{
                if (ch == '/'){
                    temp.push_back(std::stod(str));
                    str = "";
                } else if (ch == '\n'){
                    who.push_back(temp);
                    temp.clear();
                } else {
                    str += ch;
                }
            }

        }

        nt->set_weights(wih, who);


        ui->NeuralNetworkTest->setEnabled(true);
        ui->OpenImage->setEnabled(true);
        ui->UserImageCheck->setEnabled(true);
        ui->UserImagePath->setEnabled(true);
        nnEnabled = true;
}


void MainWindow::on_actionNew_triggered()
{
    delete nt;
    nt = new neuralnetwork(784, 200, 10, 0.1);


    ui->NeuralNetworkTest->setEnabled(false);
    ui->OpenImage->setEnabled(false);
    ui->UserImageCheck->setEnabled(false);
    ui->UserImagePath->setEnabled(false);
    nnEnabled = false;
}


void MainWindow::on_actionSave_triggered()
{
    if(nnEnabled){
    QString str = QFileDialog::getSaveFileName(this,
       tr("Save As"), "D:\\MySave.ntdb", tr("Neural Network (*.ntdb)"));

       //std::ofstream file;
       QFile file(str);
       if (!file.open(QIODevice::WriteOnly | QIODevice::Text))
               exit(1);

       //file.open("save.ntdb", std::ofstream::out | std::ofstream::trunc);
       QTextStream out(&file);

       for(auto &row: nt->weights_ih){
           for(auto &col: row){
               out << double(col) << "/";
           }
           out << "\n";
       }

       out << "\\";

       for(auto &row: nt->weights_ho){
           for(auto &col: row){
               out << double(col) << "/";
           }
           out << "\n";
       }
} else ui->ResultLabel->setText(tr("Neural Network isnt trained!"));
}


void MainWindow::on_NeuralNetworkTrain_clicked()
{
    ui->ResultLabel->setText("In Process..");
         QElapsedTimer timer;

    QString path = QFileDialog::getOpenFileName(this,
                                                tr("Open Train Data"), "D:\\", tr("Train Data (*.csv)"));
        QFile file(path);


        if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) { ui->ResultLabel->setText(tr("Couldnt open file!")); return;}

            char ch;

            std::vector<std::vector<long double>> inputs;
            std::vector<long double> temp;
            std::vector<int> t_marker;
            std::string str = "";
            timer.start();
            while (!file.atEnd()) {
                file.getChar(&ch);

                if (ch == '\n') {
                    temp.push_back(std::stod(str));
                    t_marker.push_back(temp[0]);
                    temp.erase(temp.begin());
                    inputs.push_back(temp);
                    temp.clear();
                    continue;
                }
                if (ch != ',')
                    str += ch;
                else {
                    temp.push_back(std::stod(str));
                    str = "";
                }
            }


            for (auto& row : inputs) {
                for (auto& col : row) {
                    col = col / 255.0 * 0.99 + 0.01;
                }
            }

            std::vector<std::vector<long double>> target;
            std::vector<long double> t_target;
            for (auto& n : t_marker) {
                for (int i = 0; i < 10; i++) if (i == n) t_target.push_back(0.99); else t_target.push_back(0.01);
                target.push_back(t_target);
                t_target.clear();
            }





            const int epochs = 10;

            for (int i = 0; i < epochs; i++) {
                for (int j = 0; j < inputs.size(); j++) {
                    nt->train(inputs[j], target[j]);
                }
            }
            ui->ResultLabel->setText(QString::fromStdString("Neural Network is ready (") + QString::number(timer.elapsed()/1000) + QString::fromStdString("s)"));


            ui->NeuralNetworkTest->setEnabled(true);
            ui->OpenImage->setEnabled(true);
            ui->UserImageCheck->setEnabled(true);
            ui->UserImagePath->setEnabled(true);
            nnEnabled = true;
}


void MainWindow::on_NeuralNetworkTest_clicked()
{

        QString path = QFileDialog::getOpenFileName(this,
                                                        tr("Open Test Data"), "D:\\", tr("Train Data (*.csv)"));
        QElapsedTimer timer;
        QFile file(path);
        std::vector<long double> temp;
        std::vector<int> t_marker;
        std::vector<std::vector<long double>> inputs;
        std::vector<long double> t_target;
        std::vector<std::vector<long double>> target;
        char ch;
        std::string str;
        timer.start();
        if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) exit(1);


            while (!file.atEnd()) {
                 file.getChar(&ch);
                if (ch == '\n') {
                    temp.push_back(std::stod(str));
                    t_marker.push_back(temp[0]);
                    temp.erase(temp.begin());
                    inputs.push_back(temp);
                    temp.clear();
                    continue;
                }
                if (ch != ',')
                    str += ch;
                else {
                    temp.push_back(std::stod(str));
                    str = "";
                }
            }


            for (auto& row : inputs) {
                for (auto& col : row) {
                    col = col / 255.0 * 0.99 + 0.01;
                }
            }


            int o_answer;
            long double tmp;
            float rate = 0, countt = 0;
            std::vector<long double> outputs;
            for (int i = 0; i < t_marker.size(); i++) {
                outputs.clear();
                outputs = nt->query(inputs[i]);

                o_answer = 0;

                tmp = 0;
                for (int j = 0; j < outputs.size(); j++) {
                    if (outputs[j] > tmp) {
                        tmp = outputs[j];
                        o_answer = j;
                    }

                }

                if (t_marker[i] == o_answer) { rate++; }
                countt++;
            }

        ui->ResultLabel->setText(QString::fromStdString("Accuracy: ") + QString::number(rate/countt) + QString::fromStdString("(") +QString::number(timer.elapsed()/1000) +QString::fromStdString("s)") );

}


void MainWindow::on_OpenImage_clicked()
{
    QString path = QFileDialog::getOpenFileName(this,
        tr("Open Picture"), "D:\\", tr("Picture (*.png)"));

        ui->UserImagePath->setText(path);
    QImage file;
    if(!file.load(ui->UserImagePath->text())) { ui->ResultLabel->setText(tr("Couldnt open file!")); return;}

    ui->Img->setPixmap(QPixmap::fromImage(file.scaled(250,250)));
}


void MainWindow::on_UserImageCheck_clicked()
{
    if(ui->UserImagePath->text() != ""){
           QImage file;
          if(!file.load(ui->UserImagePath->text())) { /* ui->ResultLabel->setText(tr("Couldnt open file!"));*/ return;}

          // ui->Img->setPixmap(QPixmap::fromImage(file.scaled(250,250)));
            file = file.scaled(28,28);
        std::vector<long double> img_data;
        int tmptmp = 0;
        for (int i = 0; i < file.height(); i++) {
            for (int j = 0; j < file.width(); j++) {
                //img_data.push_back(get_grey(img_file.pixelColor(j, i))) ;

                img_data.push_back(get_gray(file.pixelColor(j, i)));
                tmptmp++;
            }
        }

        std::vector<long double> outputs = nt->query(img_data);

        int answer = 0;
                long double tmp = 0;
                for (int i = 0; i < 10; i++) {
                    if (outputs[i] > tmp) {
                        tmp = outputs[i];
                        answer = i;
                    }
                }


                    ui->ResultLabel->setText(QString::fromStdString("NeuralNetwork says ") + QString::number(answer));
       }
}


void MainWindow::on_actionAbout_triggered()
{
    about _about(this);
    _about.setWindowModality(Qt::WindowModal);
       _about.exec();
}

