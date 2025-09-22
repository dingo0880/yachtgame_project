# 🎲 Yachtgame Project (End-to-End AI Web Service)

이 프로젝트는 단순한 미니게임이 아니라, **AI + 웹 서비스 + 배포까지 포함한 엔드투엔드 서비스** 구축 경험을 보여주기 위해 제작되었습니다.  
단순한 규칙 기반 게임을 넘어서, **Django 웹 서버, 데이터베이스, 머신러닝 기반 의사결정, 클라우드 배포**까지 풀스택 기술을 활용했습니다.  

---

## 🚀 구현 목록

### 1. 백엔드 (Django)
- 게임 룰(Yahtzee)을 서버에서 처리
- 사용자/CPU 턴 관리, 점수 계산, 보너스 로직 포함
- Django ORM을 활용한 **DB 모델링** (GameSession, TurnLog 등)

### 2. AI/머신러닝
- CPU 전략: 규칙 기반(일반형 / 공격형 / 안정형 / 랜덤형 / 엘리트형)
- **Monte Carlo Simulation** 기반 고급 CPU 전략
- **머신러닝 모델(RandomForest, LightGBM, XGBoost)** 활용  
  - 어떤 주사위를 고정할지  
  - 어떤 족보를 선택할지  
  → 실제 플레이 로그 학습 후 예측 가능

### 3. 데이터 로깅
- 모든 턴/주사위/선택 내역을 **DB + CSV**로 저장
- 커스텀 `management command`로 로그 export
- 향후 AI 학습 데이터셋으로 재사용

### 4. 프론트엔드/UX
- Django 템플릿 기반 UI 제작
- `공지사항`, `패치노트`, `게임 플레이 화면` 등 제공
- 직관적 주사위 선택 및 족보 선택 UX

### 5. 배포 (AWS Lightsail)
- Gunicorn + Nginx 조합으로 프로덕션 배포
- 정적파일 관리(`collectstatic` → `/staticfiles/`)
- 서버 reverse proxy 및 로드 설정
- 실제 **클라우드 서버에서 서비스 구동** 성공

---

## 📂 프로젝트 구조
yachtgame_project/ # Django 프로젝트 루트
yachtgame/ # 메인 앱 (뷰, 모델, ML API, 템플릿)
├─ views.py
├─ models.py
├─ ml_views.py
├─ templates/
└─ management/commands/ # 로그 export 스크립트
static/ # 개발용 정적
staticfiles/ # 배포용 정적
manage.py

---

## 🌐 실제 플레이 링크
👉 [yatch.ai.kr](http://yatch.ai.kr)

---

## 📊 기술 스택 요약
- **Backend**: Django, Django ORM  
- **Database**: SQLite (개발) / MySQL (배포)  
- **ML/AI**: RandomForest, LightGBM, XGBoost, Monte Carlo Simulation  
- **Infra/DevOps**: AWS Lightsail, Gunicorn, Nginx  
- **Frontend**: Django Template, HTML/CSS, JS  

---

## 💡 어필 포인트
- 단순한 "웹페이지"가 아니라, **데이터 수집 → 모델 학습 → 배포**까지 한 사이클 완성  
- AI/ML 요소를 실제 서비스에 **통합 적용**  
- **클라우드 서버에서 동작하는 프로덕션 서비스** 구축 경험  
- 팀 협업에 맞춘 브랜치 전략 & 커밋 컨벤션 적용  

---

## 📜 보고서 & 기술 문서
- [1편: 규칙 기반 CPU 전략](https://blog.naver.com/dingo0880/223925538295)  
- [2편: Django + MySQL + AWS 배포](https://blog.naver.com/dingo0880/223977305129)  
- [3편: 로그 수집 & 머신러닝 학습](https://blog.naver.com/dingo0880/223997922387)  

---