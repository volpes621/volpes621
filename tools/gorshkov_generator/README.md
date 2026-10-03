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
| `ship.py` | 상부구조(UKSK 블록, 함교, 마스트 하부 블록과 피라미드 탑, 연돌, 격납고와 팔라시 웰), 함수 갑판 배치, 함정 배치, 데칼(함번, 붉은 함명, 닻 격납부, 붉은 별, 함수 개구부, 선미 현창·해치, 헬기갑판 착함 구역) |
| `fittings.py` | 갑판·선체 의장품: 세로축 캡스턴, 닻줄과 스토퍼·컴프레서, V자 파도막이, 직립 원통, 소화 상자, 구명부환, 헬기갑판 안전망, 착함 통제실 |
| `weapons.py` | 무장: A-192M 130 mm 함포(평면으로 자른 볼록 다면체: 팔각 띠, 42° 앞면이 지붕 바로 아래에서 13° 윗경사로 꺾이고 삼각형 모서리 면이 있는 볼 블록, 지붕 모따기와 포신 홈 위 지붕, 큰 요가 드럼, 타공 방호링), Redut 수직발사기(셀 덮개와 힌지), UKSK 3S14, 팔라시 CIWS(받침대, 사선 보강 문이 달린 하우징, AO-18KD가 든 둥근 포드 2개와 황동 칼라·클램프 링이 있는 포신, 목 위의 광학 헤드), KT-216·장포신 기만체 발사기, MTPU 기관총 |
| `sensors.py` | 센서: 폴리멘트 배열면, 푸르케-4, 구형·캡슐형 레이돔, Pal-N 항해 레이더, 광학 지향기, ESM 상자 |
| `parts.py` | 공용 형상(둥근 상자, 파이프, 타원체, 양묘기, 체인 스토퍼, 등화 등). Slava 생성기의 해당 함수를 옮겨 왔음 |
| `markings.py` | 데칼 페인터(획으로 그리는 함번 숫자, 함명, 문, 루버, 창문 줄, 구명뗏목, 함기). Slava 생성기에서 옮겨 왔음 |
| `kit.py` | 공용 부품(갑판실, 난간, 사다리, 데칼 배치 등). Slava 생성기의 `kit.py`에서 선체 좌표와 머티리얼 이름만 바꿈 |
| `atlas.py` | 색상 팔레트(현용 러시아 해군 밝은 회청색)와 텍스처 아틀라스 레이아웃 |

`meshkit.py`(메시 프리미티브와 GLB writer), `texkit.py`(텍스처 페인팅), `ao.py`(AO 굽기)와 글꼴은 `../slava_generator`에 있는 것을 그대로 씁니다. `build.py`가 그 폴더를 import 경로에 추가합니다.

## 참고한 도면과 사진

배치와 치수는 1:500 일반배치도(측면·평면·정면, 측정용으로만 사용하고 저장소에는 넣지 않음)에서 재고, 형상은 Wikimedia Commons의 사진으로 맞췄습니다.

- Admiral Gorshkov 2018년 항공 사진(러시아 국방부, CC BY 4.0): 함수 갑판 의장품 배치, A-192M 포탑과 방호링, Redut 덮개, KT-216, 헬기갑판 표시
- Admiral Gorshkov 측면 사진(러시아 국방부, CC BY 4.0): 카메라를 맞춰 전체 윤곽과 선미 함명 위치를 대조
- Admiral Golovko 의장 공사 사진(2021, 러시아 국방부, CC BY 4.0): 닻 격납부 모양, 붉은 별, 함수 개구부, 선체 용접선
- Admiral Kasatonov 사진(2021, 러시아 국방부, CC BY 4.0): 파도막이와 캡스턴 위치 교차 확인, A-192M 포탑 앞면
- A-192M: Arsenal 공장 완성품 사진(2021, bmpd 게재)과 Arsenal 설계국 렌더(형상 참고용), Admiral Kasatonov 2020년 네바강 사진(Bestalex, CC0), Jeff Head의 Flickr 앨범에 있는 Admiral Gorshkov 사진(앞면 경사가 꺾이는 위치)
- 팔라시: airbase.ru 포럼 22350급 스레드의 Admiral Kasatonov 사진(2019년 국제해군살롱, 해군의 날), Army-2016 전시 모형 사진(Wikimedia Commons, LeAZ-1977, CC BY-SA 4.0), Palma-SU 실물 사진(러시아 국방부, CC BY 4.0)

## 좌표 규칙

- `B`는 함수 끝에서 후방으로 잰 거리(m), `x`는 좌현 방향(m), `y`는 흘수선 위 높이(m)입니다.
- 출력 좌표는 `z = 67.5 − B`이며, +Y가 위, +Z가 함수입니다. DDG-51, Slava 에셋과 같은 좌표계입니다.

## 수정 팁

- 색상은 `atlas.py`의 `PAL`에서 바꿉니다.
- 상부구조의 높이(`UKSK_Y`, `DECK01_Y`, `HANGAR_Y`, `PLATFORM_Y` 등)와 위치는 `ship.py` 상단 상수와 각 함수의 `P3(B, x, y)` 좌표로 조정합니다.
- 선체 윤곽은 `hull.py`의 `KNUCKLE`(너클 높이), `DECK_HB`(너클 반폭), `WL_HB`(흘수선 반폭), `KEEL` 표를 고칩니다. 상부 선체의 경사는 `TUMBLE`입니다.
- 팔라시와 함포는 노드 피벗이 회전축에 있으므로 엔진에서 노드를 돌려 선회·앙각을 줄 수 있습니다.
