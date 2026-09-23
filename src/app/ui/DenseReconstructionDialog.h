#ifndef DENSERECONSTRUCTIONDIALOG_H
#define DENSERECONSTRUCTIONDIALOG_H

#include <QDialog>
#include <QMap>
#include <QString>
#include <QVariant>

class QComboBox;
class ParamFormGenerator;
class QDialogButtonBox;

/**
 * @brief 密集重建方法选择对话框
 *
 * 通过 DenseReconstructionFactory 列出已注册方法，选中方法后以
 * ParamFormGenerator 展示/编辑该方法参数（schema 协议）。
 */
class DenseReconstructionDialog : public QDialog
{
    Q_OBJECT

public:
    explicit DenseReconstructionDialog(QWidget *parent = nullptr);

    QString selectedMethod() const;
    QMap<QString, QVariant> selectedParams() const;

private:
    void setupUI();
    void rebuildMethods();
    void updateParamsForCurrentMethod();

    QComboBox *m_methodCombo = nullptr;
    ParamFormGenerator *m_form = nullptr;
    QDialogButtonBox *m_buttonBox = nullptr;

    QMap<QString, QMap<QString, QVariant>> m_methodParams;
};

#endif // DENSERECONSTRUCTIONDIALOG_H