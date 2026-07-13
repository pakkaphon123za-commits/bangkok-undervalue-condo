# Shrinkage (การย่อค่า) — C vs B เปรียบเทียบแบบง่าย

เอกสารอ่านใน editor เพราะภาษาไทยแสดงใน terminal ไม่ได้

---

## แนวคิด shrinkage คืออะไร

สมมติเราอยากรู้ **slope ของแต่ละเส้นรถไฟฟ้า** (ราคาลดกี่ % ต่อกม. จากสถานี)

เรามีข้อมูลไม่เท่ากัน:

```
สุขุมวิท:        4,371 listing  →  เชื่อมาก
สีลม:           1,278 listing  →  เชื่อพอสมควร
SRT Light Red:     22 listing  →  เชื่อน้อย
```

### ปัญหาถ้าไม่มี shrinkage

SRT Light Red มีแค่ 22 listing คอนโด ถ้าบังเอิญมีคอนโดหรู 3 ตัวอยู่ใกล้สถานี
และคอนโดถูก 2 ตัวอยู่ไกล OLS อาจได้:

```
SRT Light Red slope = +0.05/km  (ราคาเพิ่มเมื่อไกลสถานี ?!)
```

มันเป็น **noise** ไม่ใช่ความจริง แต่โมเดลไม่รู้
เอาค่านี้ไปใส่ในแผนที่จะอายตาย

### Shrinkage แก้ยังไง

หลักการ: **"ข้อมูลน้อย → ดึงค่าเข้าหาค่าเฉลี่ยรวม จนกว่าจะมีหลักฐานพอ"**

```
ค่าสุดท้าย = ค่าเฉลี่ยรวม + (ค่าของเส้น − ค่าเฉลี่ยรวม) × trust_factor
```

โดย `trust_factor = n / (n + λ)`

### คำอธิบายเป็นภาพ

เปรียบเหมือน **ถามทาง 12 คน**:

- คนหนึ่งเดินผ่านเส้นนั้น 4,371 ครั้ง → **เชื่อเขาทุกคำ**
- คนหนึ่งเดินผ่านแค่ 22 ครั้ง → **ฟังเขา แต่เช็ค Google Maps ด้วย**

Shrinkage = "ฟังข้อมูล แต่ยิ่งข้อมูลน้อย ยิ่งพึ่งค่าเฉลี่ยของกลุ่ม"

---

## ตัวเลขเปรียบเทียบ (λ = 10)

| เส้น | n | trust_factor | raw slope | หลัง shrinkage |
|---|---|---|---|---|
| สุขุมวิท | 4,371 | 0.998 | −0.092 | −0.092 (แทบไม่ขยับ) |
| สีลม | 1,278 | 0.992 | −0.110 | −0.109 (ขยับนิดเดียว) |
| SRT Light Red | 22 | 0.69 | +0.050 | **−0.005** (ดึงไกลมาก) |

SRT Light Red จาก "+0.05 (ไร้สาระ)" → "−0.005 (ปลอดภัย เหมือนค่าเฉลี่ย)"

### ภาพ before/after

```
  ไม่มี shrinkage                       มี shrinkage
  ─────────────────                    ───────────────
  สุขุมวิท    █████████  −0.092        สุขุมวิท    █████████  −0.092
  สีลม        ███████████ −0.110        สีลม        ██████████ −0.109
  Blue        ████████   −0.085        Blue        ████████   −0.085
  ...
  SRT Light   ++++        +0.050       SRT Light   -           −0.005  ← แก้แล้ว!
                ↑ noise                     ↑ ดึงเข้าหาค่าปลอดภัย
```

### λ คืออะไร

λ = **ปุ่มปรับความสงสัย**

```
λ = 0    →  trust เป็น 1.0 ตลอด  →  ไม่มี shrinkage (เชื่อทุกคนเท่ากัน)
λ = 10   →  สงสัยปานกลาง        →  เส้นเล็กถูกดึงบ้าง
λ = 100  →  สงสัยสูง            →  ต้องมี ~100 listing ถึงจะถูกเชื่อ
```

เราเลือก λ เอง ค่าทั่วไป 5–20

---

## B vs C — สองเส้นทางทำ shrinkage

### ตัวเลือก B: Mixed Effects (statsmodels MixedLM)

**วิธีการ:**
- ใส่ข้อมูลทั้ง 9,599 แถวเข้าไปใน fit เดียว
- บอกโมเดลว่า "เส้นทั้งหมดมาจาก distribution เดียวกัน" (เป็นกลุ่มพี่น้อง)
- โมเดลประมาณ **ค่าเฉลี่ยรวม** + **ความแปรปรวนระหว่างเส้น** ไปพร้อมกัน
- Shrinkage เกิดอัตโนมัติผ่าน empirical Bayes — ไม่ต้องตั้ง λ เอง
- โมเดล "เรียนรู้" ว่าควรดึงขนาดไหนจากข้อมูล

**สูตรเบื้องหลัง (concept):**
```
line_slope = global_slope + random_deviation
random_deviation ~ Normal(0, σ²_between_lines)
```
σ² นี่แหละที่ทำหน้าที่ λ ใน B — แต่โมเดลประมาณเองจากข้อมูล

**ดี:**
- เป็นวิธีมาตรฐานทางสถิติ ถูกต้องที่สุด
- Shrinkage อัตโนมัติ ไม่ต้องเลือก λ
- ได้ confidence interval ที่ถูกต้อง (account for shrinkage)
- ใน portfolio ดูเป็นมืออาชีพ

**เสีย:**
- ต้องลง statsmodels (`pip install --user --break-system-packages statsmodels`)
  — ของไม่ได้ลงอยู่ตอนนี้
- **อาจ converge ไม่ออก** — MixedLM ใช้ maximum likelihood optimization
  กับ sparse groups (SRT n=22) มักมีปัญหา อาจเจอ `ConvergenceWarning`
  หรือ fit ไม่สำเร็จเลย
- Output อ่านยาก: fixed effects + random effects covariance matrix + BLUPs
  — ต้องอธิบายยาว
- Debug ยากเวลามีปัญหา
- โอเวอร์คิลล์ถ้าเราจะ filter เส้นที่ n < 100 ออกจากผลลัพธ์อยู่ดี

---

### ตัวเลือก C: OLS รายเส้น + shrinkage มือ

**วิธีการ (2 ขั้น):**

**ขั้นที่ 1 — fit เส้นละ OLS:**
- แบ่ง 9,599 แถวเป็น ~12 กลุ่มตามเส้น
- รัน OLS เล็กๆ แต่ละกลุ่ม: `log(price) ~ distance`
- ได้ raw intercept + raw slope ของแต่ละเส้น (อิสระต่อกัน)

**ขั้นที่ 2 — ดึงเข้าหาค่าเฉลี่ย:**
- คำนวณ global mean ของ raw slopes และ intercepts
- แต่ละเส้น:
  ```
  shrunk = global_mean + (raw − global_mean) × n / (n + λ)
  ```
- λ เป็นค่าที่เราเลือกเอง (เช่น 10)

**สูตรเบื้องหลัง:**
```
trust_factor = n / (n + λ)
estimate_line = global_mean + (raw_line − global_mean) × trust_factor
```

**ดี:**
- **ไม่ต้องลง dependency ใหม่** — ใช้ numpy/pandas ที่มีอยู่
- โปร่งใสมาก — พิมพ์ raw vs shrunk เปรียบเทียบเป็นตาราง เห็น shrinkage เกิดจริง
- ไม่มี convergence failure เป็นไปไม่ได้ (OLS แต่ละเส้น fit เสมอ)
- λ คุมได้ — ทดสอบ sensitivity ได้ (λ=5 vs λ=20)
- ตรงกับสูตรใน questions.md ที่เราอ่านมาเป๊ะ

**เสีย:**
- เป็น approximation ของ B — uncertainty estimates หลัง shrinkage
  ไม่ถูกต้อง 100% (ต้องใช้ delta method ถึงจะถูก)
- เลือก λ เอง subjective — ต้อง justify ใน writeup
- สถิติแท้จะถาม "ทำไมไม่ใช้ mixed model ไปเลย?"

---

## เปรียบเทียบเคียงข้างกัน

| ด้าน | B: MixedLM | C: OLS + shrinkage มือ |
|---|---|---|
| Shrinkage อัตโนมัติ? | ใช่ | ไม่ ใช้สูตร |
| ต้องลง dependency? | ใช่ (statsmodels) | ไม่ |
| Risk converge ล้มเหลว? | มี (สูงกับ sparse group) | ไม่มี |
| λ ใครเลือก? | โมเดลประมาณเอง | เราเลือกเอง |
| Confidence interval ถูก? | ถูก | ประมาณการ |
| Output เข้าใจง่าย? | ยาก (fixed + random effects) | ง่าย (raw vs shrunk ตาราง) |
| Debug ยากไหม? | ยาก | ง่าย |
| Portfolio ดูเป็น? | มืออาชีพ | โปร่งใส อธิบายได้ |

---

## ความแตกต่างในผลลัพธ์จริง (สำหรับ project นี้)

สมมติเราเทียบ slope ของสุขุมวิท:

```
B (MixedLM):     −0.0921  ± 0.003
C (OLS + λ=10):  −0.0920  ± 0.004 (approx)
```

ต่างกันที่ทศนิยมตัวที่ 4 — **คนดูแผนที่/undervalued flags ไม่มีทางบอกได้**

ส่วนเส้น sparse (SRT Light Red n=22):

```
B:  −0.008  ± 0.020  (shrinkage อัตโนมัติ, CI ถูก)
C:  −0.005  ± 0.025  (shrinkage มือ, CI ประมาณ)
```

ทั้งคู่ "ดึงเข้าหาเฉลี่ย" ผลใกล้กันมาก แต่ B ให้ uncertainty ที่ถูกต้องกว่า

---

## สรุป: ทำไม C พอเพียงสำหรับ project นี้

project เรา output คือ:
1. Decay curve บนแผนที่ (visual)
2. Undervalued flag (z-score threshold)
3. LLM narrative (text)

B กับ C ให้ผลเหมือนกัน 95% ใน output ที่ผู้ใช้เห็น ส่วนต่าง 5% คือ:
- B: confidence interval ถูกต้องกว่า
- B: λ ประมาณจากข้อมูล ไม่ต้อง justify
- B: ดูเป็นมืออาชีพกว่าใน academic context

แต่ราคาที่จ่าย:
- ลง statsmodels (อาจ conflict กับ numpy 2.5)
- เสี่ยง convergence failure แล้วต้อง debug
- output อธิบายยากขึ้น

C เป็น approximation ที่:
- ไม่มี risk เลย
- โปร่งใส พิมพ์ raw vs shrunk ให้เห็น
- ตรงกับที่อ่านมาใน questions.md
- ถ้าอยากอัปเกรดเป็น B ทีหลัง แค่แทนขั้นที่ 2 ขั้นที่ 1 ใช้ต่อได้

---

## สรุปสั้น

**B (MixedLM)** = โรงงานผลิต ถูกต้องที่สุด แต่ต้องขับรถไปรับ (ลง statsmodels)
และบางครั้งสตาร์ทไม่ติด (convergence failure)

**C (OLS + shrinkage มือ)** = ชุดประกอบ ทำเอง ผลใกล้เคียง ไม่มีความเสี่ยง
เห็นทุกขั้นตอน ตรงกับที่เรียนมาใน questions.md

สำหรับ project นี้ (portfolio + แผนที่ + flags) C พอ
สำหรับงานวิชาการ/review เข้มงวด B คุ้มกว่า

---

## คำถามก่อนตัดสินใจ

ก่อนเลือกระหว่าง B กับ C มี 2 คำถามต้องตอบ:

1. **ยอมลง statsmodels และจัดการ convergence issue ไหม?**
   ใช่ → B ถูกต้องกว่า / ไม่ → C

2. **คอนโด interchange (31 listing ที่ ARL+สุขุมวิท, 27 ที่ Blue+Purple)
   จัดยังไง?**
   - ให้เส้นแรก
   - ทำซ้ำทั้งสองเส้น
   - แยกเป็นหมวด interchange
