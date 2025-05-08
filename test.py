import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------
# NumPy 기초
# ---------------------------

# 1차원 배열 생성
arr1 = np.array([1, 2, 3, 4])
print("1차원 배열:", arr1)

# 2차원 배열 생성
arr2 = np.array([[1, 2], [3, 4]])
print("2차원 배열:\n", arr2)

# 배열의 형태(shape)와 차원(ndim)
print("arr2의 형태:", arr2.shape)
print("arr2의 차원 수:", arr2.ndim)

# 기본 연산 (원소별 연산)
print("각 원소에 2를 곱함:", arr1 * 2)

# 브로드캐스팅 예시
arr3 = np.array([[1], [2], [3]])
print("브로드캐스팅 결과:\n", arr3 + arr1)

# ---------------------------
# pandas 기초
# ---------------------------

# 시리즈 생성
s = pd.Series([10, 20, 30], index=["a", "b", "c"])
print("pandas Series:\n", s)

# 데이터프레임 생성
data = {
    "이름": ["민호", "지수", "철수"],
    "나이": [23, 25, 21],
    "전공": ["컴퓨터", "통계", "물리"]
}
df = pd.DataFrame(data)
print("pandas DataFrame:\n", df)

# 특정 열 접근
print("나이 열:\n", df["나이"])

# 조건 필터링
print("나이가 23 이상인 사람:\n", df[df["나이"] >= 23])

# 새로운 열 추가
df["졸업 여부"] = ["미정", "졸업", "미정"]
print("새로운 열 추가 후:\n", df)

# ---------------------------
# matplotlib 기초
# ---------------------------

# 라인 그래프 예시: NumPy 배열 활용
x = np.linspace(0, 10, 100)  # 0부터 10까지 100개의 값
y = np.sin(x)
plt.figure()
plt.plot(x, y)
plt.title("Sine Curve")
plt.xlabel("x")
plt.ylabel("sin(x)")
plt.grid(True)
plt.show()

# 막대 그래프 예시: pandas Series 활용
score_series = pd.Series([85, 90, 78], index=["국어", "수학", "영어"])
plt.figure()
score_series.plot(kind="bar", title="과목별 점수")
plt.ylabel("점수")
plt.show()

# 히스토그램 예시: 무작위 데이터 분포
random_data = np.random.randn(1000)
plt.figure()
plt.hist(random_data, bins=30, alpha=0.7, color='g')
plt.title("정규 분포 히스토그램")
plt.xlabel("값")
plt.ylabel("빈도")
plt.show()