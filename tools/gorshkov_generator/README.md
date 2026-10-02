# gorshkov_generator

`assets/models/admiral_gorshkov_class_frigate.glb`를 만드는 절차적 모델링 스크립트입니다. Slava 생성기와 같은 방식으로, 외부 3D 툴 없이 Python만으로 선체, 상부구조, 무장, 텍스처 아틀라스, AO 굽기, GLB 출력까지 처리합니다. 같은 입력이면 항상 같은 파일이 바이트 단위로 똑같이 나옵니다.

## 실행

```bash
pip install -r requirements.txt          # embreex는 선택: 설치하면 AO 굽기가 빨라짐
python build.py out.glb                  # Admiral Gorshkov 454 (기본)
python build.py out.glb --ship kasatonov # Admiral Kasatonov 461
python build.py out.glb --ship golovko   # Admiral Golovko 456
python build.py out.glb --pennant 417    # 함번만 바꾸기 (Admiral Gorshkov의 2018~2020년 함번)
python build.py out.glb --no-ao          # AO 굽기 생략(빠름)
```

## 파일 구성

| 파일 | 내용 |
|---|---|
| `build.py` | 실행 진입점과 함정 버전 정의 |
| `hull.py` | 선체 형상. 1:500 일반배치도에서 측정한 선도 표(너클 높이, 반폭, 흘수선, 용골선)와 소나 돔 치수 |
| `model.py` | 선체·상부 선체(5.5° 안쪽 경사)·함수 현장·갑판·헬기갑판 메시, 소나 돔, 추진축·프로펠러·방향타, 선체/갑판 텍스처 밴드, 머티리얼, GLB 출력 |
| `ship.py` | 상부구조(UKSK 블록, 함교, 마스트 하부 블록과 피라미드 탑, 연돌, 격납고와 팔라시 웰), 함수 의장품, 함정 배치, 데칼(함번·함명·헬기갑판) |
| `weapons.py` | 무장: A-192M 130 mm 함포, Redut 수직발사기, UKSK 3S14, 팔라시 CIWS, KT-216 기만체 발사기, MTPU 기관총 |
| `sensors.py` | 센서: 폴리멘트 배열면, 푸르케-4, 구형·캡슐형 레이돔, Pal-N 항해 레이더, 광학 지향기, ESM 상자 |
| `parts.py` | 공용 형상(둥근 상자, 파이프, 타원체, 양묘기, 체인 스토퍼, 등화 등). Slava 생성기의 해당 함수를 옮겨 왔음 |
| `markings.py` | 데칼 페인터(획으로 그리는 함번 숫자, 함명, 문, 루버, 창문 줄, 구명뗏목, 함기). Slava 생성기에서 옮겨 왔음 |
| `kit.py` | 공용 부품(갑판실, 난간, 사다리, 데칼 배치 등). Slava 생성기의 `kit.py`에서 선체 좌표와 머티리얼 이름만 바꿈 |
| `atlas.py` | 색상 팔레트(현용 러시아 해군 밝은 회청색)와 텍스처 아틀라스 레이아웃 |

`meshkit.py`(메시 프리미티브와 GLB writer), `texkit.py`(텍스처 페인팅), `ao.py`(AO 굽기)와 글꼴은 `../slava_generator`에 있는 것을 그대로 씁니다. `build.py`가 그 폴더를 import 경로에 추가합니다.

## 좌표 규칙

- `B`는 함수 끝에서 후방으로 잰 거리(m), `x`는 좌현 방향(m), `y`는 흘수선 위 높이(m)입니다.
- 출력 좌표는 `z = 67.5 − B`이며, +Y가 위, +Z가 함수입니다. DDG-51, Slava 에셋과 같은 좌표계입니다.

## 수정 팁

- 색상은 `atlas.py`의 `PAL`에서 바꿉니다.
- 상부구조의 높이(`UKSK_Y`, `DECK01_Y`, `HANGAR_Y`, `PLATFORM_Y` 등)와 위치는 `ship.py` 상단 상수와 각 함수의 `P3(B, x, y)` 좌표로 조정합니다.
- 선체 윤곽은 `hull.py`의 `KNUCKLE`(너클 높이), `DECK_HB`(너클 반폭), `WL_HB`(흘수선 반폭), `KEEL` 표를 고칩니다. 상부 선체의 경사는 `TUMBLE`입니다.
- 팔라시와 함포는 노드 피벗이 회전축에 있으므로 엔진에서 노드를 돌려 선회·앙각을 줄 수 있습니다.
