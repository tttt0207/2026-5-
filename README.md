# 发挥第五题串口通信与 STM32 逻辑说明

本文档用于说明修改后的**发挥部分第五题**中，香橙派视觉端与 STM32 算法端之间的串口通信流程、状态机逻辑以及收发数据格式。

## 1. 修改背景

发挥第五题的正确逻辑是：

> 先识别目标物；当目标物大致处于画面中心后，香橙派通知 STM32“目标物已找到”；STM32 端短暂停止云台一段时间后继续转动；后续香橙派继续按照原逻辑寻找空白画板，找到黑框后让 STM32 停止扫描并进入绘图流程。

因此，第五题中香橙派需要完成两次视觉判断：

1. **目标物识别与居中判断**
2. **空白画板黑框识别与居中判断**

STM32 端需要区分这两个阶段对应的控制帧。

---

## 2. 串口基础参数

| 项目 | 参数 |
|---|---|
| 波特率 | 115200 |
| 数据格式 | 8N1 |
| 香橙派串口 | `/dev/ttyS2` |
| 发送周期 | 50 ms |
| 单帧长度 | 6 字节 |
| 电平 | 3.3V TTL |
| 接线 | TX-RX、RX-TX、GND-GND |

---

## 3. 坐标系与 dx/dy 含义

香橙派端使用画板坐标系：

- 画板中心为原点
- x 轴向右为正
- y 轴向上为正
- 单位为 cm

当前激光点使用画面中心十字映射到画板坐标系得到。

```text
dx = 目标点 x 坐标 - 当前画面中心 x 坐标
dy = 目标点 y 坐标 - 当前画面中心 y 坐标
```

dx/dy 发送时扩大 100 倍，转换为 int16：

```c
dx_i = round(dx_cm * 100);
dy_i = round(dy_cm * 100);
```

STM32 端还原方式：

```c
int16_t dx_raw = (int16_t)((buf[1] << 8) | buf[2]);
int16_t dy_raw = (int16_t)((buf[3] << 8) | buf[4]);

float dx_cm = dx_raw / 100.0f;
float dy_cm = dy_raw / 100.0f;
```

---

## 4. 发挥第五题完整流程

```text
STM32 发送状态 25 启动帧
        ↓
STM32 云台慢速扫描
        ↓
香橙派识别目标物
        ↓
目标物大致居中后，香橙派发送“目标物已找到”
        ↓
STM32 停止云台一段时间
        ↓
STM32 继续慢速扫描
        ↓
香橙派寻找空白画板黑框
        ↓
空白画板大致居中后，香橙派发送“停止扫描”
        ↓
STM32 停止扫描并等待云台停稳
        ↓
STM32 发送“画板 ready”
        ↓
香橙派发送移动到起点数据
        ↓
到达起点并稳定停留一段时间
        ↓
香橙派请求开激光
        ↓
STM32 打开激光并回 ACK
        ↓
香橙派正式发送巡线数据
        ↓
轨迹完成后，香橙派请求关激光
        ↓
STM32 关闭激光并回 ACK
        ↓
任务完成
```

---

## 5. 状态 25 使用的串口帧

### 5.1 STM32 启动第五题

方向：

```text
STM32 -> 香橙派
```

数据：

```text
AA 25 00 00 DA 55
```

含义：

```text
启动发挥第五题
```

其中：

```text
check = state ^ 0xFF
0x25 ^ 0xFF = 0xDA
```

---

### 5.2 目标物已找到

方向：

```text
香橙派 -> STM32
```

数据：

```text
03 05 00 00 00 30
```

含义：

```text
目标物已经被识别到，并且已经大致处于画面中心
```

STM32 收到后应执行：

```text
1. 停止云台转动
2. 保持一小段时间
3. 继续慢速扫描，寻找空白画板
```

注意：

- 该帧不需要 STM32 回 ACK。
- 该帧只表示“目标物已找到”，不是“画板已找到”。
- 不能收到该帧后直接进入绘图流程。

---

### 5.3 空白画板已找到，请求停止扫描

方向：

```text
香橙派 -> STM32
```

数据：

```text
03 03 00 00 00 30
```

含义：

```text
空白画板黑框已经大致位于画面中心，请 STM32 停止扫描
```

STM32 收到后应执行：

```text
1. 停止云台扫描
2. 等待云台停稳
3. 发送画板 ready 帧
```

---

### 5.4 画板 ready

方向：

```text
STM32 -> 香橙派
```

数据：

```text
03 04 00 00 00 30
```

含义：

```text
画板已经进入可绘制视野，云台已经停稳，香橙派可以开始移动到起点
```

STM32 可以每 50 ms 重复发送该帧，直到收到香橙派发来的移动到起点帧：

```text
02 dxH dxL dyH dyL 04
```

---

### 5.5 移动到起点

方向：

```text
香橙派 -> STM32
```

数据格式：

```text
02 dxH dxL dyH dyL 04
```

含义：

```text
空走到轨迹起点
```

STM32 收到后应执行：

```text
1. 激光保持关闭
2. 根据 dx/dy 控制云台移动
3. 不需要自己判断是否到达起点
```

到达起点的判断由香橙派完成。

---

### 5.6 请求开激光 / 开激光 ACK

方向 1：

```text
香橙派 -> STM32
```

方向 2：

```text
STM32 -> 香橙派
```

数据相同：

```text
03 01 00 00 00 30
```

含义：

```text
请求打开激光 / 确认激光已经打开
```

注意：

香橙派已经加入起点稳定判断：

```text
只有当画面中心到达起点，并且在允许误差范围内稳定停留一段时间后，才会发送开激光请求。
```

STM32 收到后应执行：

```text
1. 打开激光
2. 回发 03 01 00 00 00 30 作为 ACK
```

---

### 5.7 正式画图巡线

方向：

```text
香橙派 -> STM32
```

数据格式：

```text
01 dxH dxL dyH dyL 05
```

含义：

```text
正式画图阶段的 dx/dy 控制数据
```

STM32 收到后应执行：

```text
1. 保持激光打开
2. 根据 dx/dy 控制云台
3. 让当前光斑跟随目标轨迹点
```

---

### 5.8 请求关激光 / 关激光 ACK

方向 1：

```text
香橙派 -> STM32
```

方向 2：

```text
STM32 -> 香橙派
```

数据相同：

```text
03 02 00 00 00 30
```

含义：

```text
请求关闭激光 / 确认激光已经关闭
```

STM32 收到后应执行：

```text
1. 关闭激光
2. 回发 03 02 00 00 00 30 作为 ACK
3. 当前题目结束
```

---

## 6. 第五题 STM32 推荐状态机

推荐将第五题单独写成清晰的状态机：

```c
typedef enum
{
    MCU_IDLE = 0,

    MCU_SEND_TASK25_CMD,

    MCU_SCAN_TARGET,
    MCU_TARGET_PAUSE,

    MCU_SCAN_BOARD,
    MCU_WAIT_BOARD_STABLE,
    MCU_SEND_BOARD_READY,

    MCU_MOVE_TO_START,
    MCU_WAIT_LASER_ON,
    MCU_DRAWING,
    MCU_WAIT_LASER_OFF,

    MCU_TASK_DONE
} MCU_State_t;
```

| 状态 | 含义 |
|---|---|
| `MCU_IDLE` | 空闲，等待按键 |
| `MCU_SEND_TASK25_CMD` | 发送第五题启动帧 |
| `MCU_SCAN_TARGET` | 慢速扫描，等待目标物进入画面中心 |
| `MCU_TARGET_PAUSE` | 收到目标物已找到后，短暂停留 |
| `MCU_SCAN_BOARD` | 继续扫描，寻找空白画板 |
| `MCU_WAIT_BOARD_STABLE` | 收到停止扫描后，等待云台停稳 |
| `MCU_SEND_BOARD_READY` | 向香橙派发送画板 ready |
| `MCU_MOVE_TO_START` | 接收 0x02 帧，空走到起点 |
| `MCU_WAIT_LASER_ON` | 等待香橙派请求开激光 |
| `MCU_DRAWING` | 接收 0x01 帧，正式巡线 |
| `MCU_WAIT_LASER_OFF` | 等待香橙派请求关激光 |
| `MCU_TASK_DONE` | 任务完成，回到空闲 |

---

## 7. STM32 端接收解析建议

所有帧都是 6 字节，建议使用滑动窗口解析。

### 7.1 控制帧判断

```c
if (buf[0] == 0x03 && buf[5] == 0x30)
{
    if (buf[1] == 0x05)
    {
        // 目标物已找到
        // 只在 MCU_SCAN_TARGET 阶段处理
    }
    else if (buf[1] == 0x03)
    {
        // 空白画板已找到，请求停止扫描
        // 只在 MCU_SCAN_BOARD 阶段处理
    }
    else if (buf[1] == 0x01)
    {
        // 请求开激光
    }
    else if (buf[1] == 0x02)
    {
        // 请求关激光
    }
}
```

### 7.2 移动到起点帧判断

```c
if (buf[0] == 0x02 && buf[5] == 0x04)
{
    int16_t dx_raw = (int16_t)((buf[1] << 8) | buf[2]);
    int16_t dy_raw = (int16_t)((buf[3] << 8) | buf[4]);

    float dx_cm = dx_raw / 100.0f;
    float dy_cm = dy_raw / 100.0f;

    laser_off();
    use_pid_to_move(dx_cm, dy_cm);
}
```

### 7.3 正式画图帧判断

```c
if (buf[0] == 0x01 && buf[5] == 0x05)
{
    int16_t dx_raw = (int16_t)((buf[1] << 8) | buf[2]);
    int16_t dy_raw = (int16_t)((buf[3] << 8) | buf[4]);

    float dx_cm = dx_raw / 100.0f;
    float dy_cm = dy_raw / 100.0f;

    laser_on_keep();
    use_pid_to_move(dx_cm, dy_cm);
}
```

---

## 8. 第五题关键区别

第五题中最容易混淆的是这两个控制帧：

| 数据 | 含义 | STM32 行为 |
|---|---|---|
| `03 05 00 00 00 30` | 目标物已找到 | 停一下，然后继续扫描 |
| `03 03 00 00 00 30` | 空白画板已找到 | 停止扫描，停稳后发 ready |

一定不要把 `03 05` 当成停止扫描画板的指令。

---

## 9. STM32 端第五题简化伪代码

```c
switch (mcu_state)
{
case MCU_SEND_TASK25_CMD:
    send_frame(0xAA, 0x25, 0x00, 0x00, 0xDA, 0x55);
    start_slow_scan();
    mcu_state = MCU_SCAN_TARGET;
    break;

case MCU_SCAN_TARGET:
    if (recv_frame_is(0x03, 0x05, 0x00, 0x00, 0x00, 0x30))
    {
        stop_scan();
        pause_start_time = millis();
        mcu_state = MCU_TARGET_PAUSE;
    }
    break;

case MCU_TARGET_PAUSE:
    if (millis() - pause_start_time >= TARGET_PAUSE_TIME_MS)
    {
        start_slow_scan();
        mcu_state = MCU_SCAN_BOARD;
    }
    break;

case MCU_SCAN_BOARD:
    if (recv_frame_is(0x03, 0x03, 0x00, 0x00, 0x00, 0x30))
    {
        stop_scan();
        stable_start_time = millis();
        mcu_state = MCU_WAIT_BOARD_STABLE;
    }
    break;

case MCU_WAIT_BOARD_STABLE:
    if (millis() - stable_start_time >= BOARD_STABLE_TIME_MS)
    {
        mcu_state = MCU_SEND_BOARD_READY;
    }
    break;

case MCU_SEND_BOARD_READY:
    send_frame(0x03, 0x04, 0x00, 0x00, 0x00, 0x30);

    if (recv_head_tail(0x02, 0x04))
    {
        mcu_state = MCU_MOVE_TO_START;
    }
    break;

case MCU_MOVE_TO_START:
    if (recv_head_tail(0x02, 0x04))
    {
        laser_off();
        parse_dx_dy_and_pid_control();
    }

    if (recv_frame_is(0x03, 0x01, 0x00, 0x00, 0x00, 0x30))
    {
        laser_on();
        send_frame(0x03, 0x01, 0x00, 0x00, 0x00, 0x30);
        mcu_state = MCU_DRAWING;
    }
    break;

case MCU_DRAWING:
    if (recv_head_tail(0x01, 0x05))
    {
        laser_on_keep();
        parse_dx_dy_and_pid_control();
    }

    if (recv_frame_is(0x03, 0x02, 0x00, 0x00, 0x00, 0x30))
    {
        laser_off();
        send_frame(0x03, 0x02, 0x00, 0x00, 0x00, 0x30);
        mcu_state = MCU_TASK_DONE;
    }
    break;

case MCU_TASK_DONE:
    stop_scan();
    laser_off();
    mcu_state = MCU_IDLE;
    break;

default:
    mcu_state = MCU_IDLE;
    break;
}
```

---

## 10. 调试建议

建议按下面顺序测试：

1. STM32 发送 `AA 25 00 00 DA 55`，确认香橙派进入状态 25。
2. 云台慢速扫描，目标物进入画面中心后，确认 STM32 能收到：

   ```text
   03 05 00 00 00 30
   ```

3. STM32 收到 `03 05` 后，确认云台会停一下，然后继续转动。
4. 空白画板进入画面中心后，确认 STM32 能收到：

   ```text
   03 03 00 00 00 30
   ```

5. STM32 收到 `03 03` 后，确认云台停止，并发送：

   ```text
   03 04 00 00 00 30
   ```

6. 香橙派收到 ready 后，应开始发送：

   ```text
   02 dxH dxL dyH dyL 04
   ```

7. 到起点稳定后，香橙派发送：

   ```text
   03 01 00 00 00 30
   ```

8. STM32 打开激光并回发：

   ```text
   03 01 00 00 00 30
   ```

9. 香橙派开始正式发送：

   ```text
   01 dxH dxL dyH dyL 05
   ```

10. 结束后，香橙派发送：

    ```text
    03 02 00 00 00 30
    ```

11. STM32 关闭激光并回发：

    ```text
    03 02 00 00 00 30
    ```

---

## 11. 最终速查表

| 功能 | 方向 | 数据 |
|---|---|---|
| 启动第五题 | STM32 -> 香橙派 | `AA 25 00 00 DA 55` |
| 目标物已找到 | 香橙派 -> STM32 | `03 05 00 00 00 30` |
| 空白画板已找到，请求停止扫描 | 香橙派 -> STM32 | `03 03 00 00 00 30` |
| 画板 ready | STM32 -> 香橙派 | `03 04 00 00 00 30` |
| 移动到起点 | 香橙派 -> STM32 | `02 dxH dxL dyH dyL 04` |
| 请求/确认开激光 | 双向 | `03 01 00 00 00 30` |
| 正式画图 | 香橙派 -> STM32 | `01 dxH dxL dyH dyL 05` |
| 请求/确认关激光 | 双向 | `03 02 00 00 00 30` |

---

## 12. 注意事项

1. `03 05 00 00 00 30` 只表示目标物已找到，不表示画板已找到。
2. 收到 `03 05` 后，STM32 应该短暂停留，然后继续扫描。
3. 收到 `03 03` 后，STM32 才应该真正停止扫描并等待画板稳定。
4. `0x02` 阶段必须强制关闭激光。
5. 只有收到 `03 01 00 00 00 30` 后才能打开激光。
6. 香橙派已经加入“起点稳定停留后再开激光”的判断，STM32 不需要判断是否到达起点。
7. STM32 收到开激光请求后，应先打开激光，再回 ACK。
8. STM32 收到关激光请求后，应先关闭激光，再回 ACK。

