# slava_generator

`assets/models/slava_class_cruiser.glb`를 만드는 절차적 모델링 스크립트입니다. 외부 3D 툴 없이 Python만으로 선체, 상부구조, 무장, 텍스처 아틀라스, AO 굽기, GLB 출력까지 모두 처리합니다. 같은 입력이면 항상 같은 파일이 바이트 단위로 똑같이 나옵니다.

## 실행

```bash
pip install -r requirements.txt          # embreex는 선택: 설치하면 AO 굽기가 빨라짐
python build.py out.glb                  # Moskva 121 (기본)
python build.py out.glb --ship varyag    # Varyag 011
python build.py out.glb --ship ustinov   # Marshal Ustinov 055
python build.py out.glb --pennant 108    # 함번만 바꾸기
python build.py out.glb --no-ao          # AO 굽기 생략(빠름)
```

## 파일 구성

| 파일 | 내용 |
|---|---|
| `build.py` | 실행 진입점과 함정 버전 정의 |
| `hull.py` | 선체 형상. Kuleshov 도면에서 측정한 선도 표를 담고 있음 |
| `model.py` | 선체·갑판 메시, 선체/갑판 텍스처 밴드 페인팅, 머티리얼, GLB 출력 |
| `ship.py` | 상부구조, 무장, 센서, 마스트, 보트, 헬기갑판, 데칼(함번·함명·별). 함번 숫자는 사진의 서체를 따라 획으로 그림 |
| `weapons.py` | 무장: AK-130 포탑과 포신, AK-630M(6연장 포신 다발), S-300F 해치(셀 덮개 8개, 덮개마다 작은 원 3개)·구동부·장전 갠트리, RBU-6000, PK-2 |
| `detail.py` | 세부 장비: MR-123·MR-184·Pop Group 사격통제 레이더, Top Dome, 발사기 받침대, 굴뚝 배기구, 함재정과 대빗, 크레인 |
| `antennas.py` | 탐색 레이더 안테나(Top Pair, Fregat, Front Door): 곡면 반사판, 안테나별 격자 텍스처, 급전부, 뒷면 트러스 |
| `deckgear.py` | 갑판 장비: AK-130 타공 방폭판, 환기구, 보관함, 양묘기, 체인 스토퍼, 캡스턴, 항해등, 레이돔 |
| `kit.py` | 공용 부품(갑판실, 난간, 사다리, 격자 패널, 트러스, 구명뗏목, 알 모양 ECM 레이돔 등) |
| `atlas.py` | 색상 팔레트와 텍스처 아틀라스 레이아웃(밴드, 견본색) |
| `meshkit.py` | 메시 프리미티브(박스, 원기둥, 회전체, 로프트, 튜브)와 GLB writer |
| `texkit.py` | 텍스처 페인팅 유틸리티(노이즈, 노멀맵 생성) |
| `ao.py` | 정점 AO 굽기(Embree 레이캐스트) |

`meshkit.py`, `texkit.py`, `ao.py`와 `fonts/`는 `tools/gorshkov_generator`도 그대로 가져다 씁니다. 이 파일들을 고친 뒤에는 두 모델을 모두 다시 생성해 결과를 확인하세요.

## 좌표 규칙

- 도면 기준으로 `B`는 함수 끝에서 후방으로 잰 거리(m), `x`는 좌현 방향(m), `y`는 흘수선 위 높이(m)입니다.
- 출력 좌표는 `z = 93.2 − B`이며, +Y가 위, +Z가 함수입니다. DDG-51 에셋과 같은 좌표계입니다.

## 수정 팁

- 색상은 `atlas.py`의 `PAL`에서 바꿉니다. 예를 들어 `deck_red`를 회색으로 바꾸면 2012년 Moskva처럼 회색 갑판이 됩니다.
- 장비 위치는 `ship.py` 각 함수의 `P3(B, x, y)` 좌표로 조정합니다.
- 삼각형 수는 원기둥·회전체의 `seg` 값과 `ship.AO['max_len']`(AO용 면 분할 크기)으로 조절합니다.
