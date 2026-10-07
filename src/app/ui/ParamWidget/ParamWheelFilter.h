#ifndef PARAMWHEELFILTER_H
#define PARAMWHEELFILTER_H

#include <QEvent>
#include <QObject>
#include <QWidget>

/**
 * @brief 参数控件滚轮过滤器（焦点门控）。
 *
 * 问题：QSpinBox / QDoubleSpinBox / QComboBox 默认在光标悬停时响应滚轮
 * 修改数值，与可滚动的稀疏/密集重建对话框冲突（想滚动页面却改了参数）。
 *
 * 行为：
 *  - 控件无焦点：滚轮不修改值，事件标记 ignore 并停止控件默认处理，
 *    交还 Qt 的滚轮传播链由外层 QScrollArea 滚动页面；
 *  - 控件持有焦点（用户点入控件）：滚轮照常调值（主动交互）。
 *
 * 用法：editor->installEventFilter(new ParamWheelFilter(editor))，
 * 以编辑器为 parent 随其销毁。
 *
 * ⚠ 必须同时把控件 focusPolicy 设为 StrongFocus（QSpinBox/QComboBox
 * 默认 WheelFocus——窗口层会在滚轮事件送达前把焦点授予悬停控件，
 * 导致本过滤器的 hasFocus 判断失效、悬停照样改值）。
 */
class ParamWheelFilter : public QObject
{
public:
    explicit ParamWheelFilter(QObject *parent) : QObject(parent) {}

    bool eventFilter(QObject *watched, QEvent *event) override
    {
        if (event->type() == QEvent::Wheel) {
            auto *w = qobject_cast<QWidget *>(watched);
            if (w && !w->hasFocus()) {
                // 注意：QWheelEvent 默认 accepted，先 ignore 使 notify 的
                // 滚轮传播链继续向上（直至 QAbstractScrollArea 滚动页面）
                event->ignore();
                return true;
            }
        }
        return QObject::eventFilter(watched, event);
    }
};

#endif // PARAMWHEELFILTER_H
