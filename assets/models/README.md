# Slava급 순양함 (Project 1164) — 로우폴리 모델

![bow](previews/slava_bow.png)

`slava_class_cruiser.glb`는 Free DDG-51(Arleigh Burke) 에셋과 같은 화풍, 같은 스케일로 새로 만든 Slava급 순양함 모델입니다. 기존 Sketchfab "Russian Slava-class cruiser" 모델은 품질이 낮아 지오메트리와 텍스처를 하나도 재사용하지 않고 처음부터 다시 만들었습니다. 기본 표기는 **Moskva(함번 121)** 입니다.

| 항목 | Slava (이 모델) | DDG-51 (기준 에셋) |
|---|---|---|
| 형식 | glTF 2.0 바이너리(`.glb`), 확장 없음 | glTF 2.0 바이너리 |
| 삼각형 / 정점 | 44,111 / 40,567 | 49,381 / 59,640 |
| 머티리얼 | 3개 | 5개 |
| 텍스처 구성 | baseColor + metallicRoughness + normal | 동일 |
| alpha 모드 | MASK(난간·레이더 메시·안전망) | MASK |
| 파일 크기 | 5.4 MB | 5.4 MB |
| 전장 | 186.4 m (실측 1:1) | 153.9 m (실측 1:1) |

## 좌표계

DDG-51 에셋과 같습니다. 같은 씬에 그대로 배치하면 실제 크기 비율이 맞습니다.

- 단위는 미터입니다.
- +Y가 위, +Z가 함수(뱃머리), +X가 좌현입니다.
- 설계 흘수선은 y = 0입니다.
- 원점은 전장의 중앙(z = 0)입니다. 함수는 z = +93.2, 함미는 z = −93.2입니다.

![vs DDG-51](previews/slava_vs_ddg51.png)

## 머티리얼

| 이름 | 크기 | 용도 |
|---|---|---|
| `Slava_Atlas` | 2048² RGBA | 선체 측면, 갑판 도장, 함번·함명, 창문, 난간, 레이더 메시, 헬기갑판 마킹, 장비 색상 견본 |
| `Slava_Paint` | 1024² 타일 | 상부구조와 장비의 회청색 도장 |
| `Slava_Deck` | 512² 타일 | 적갈색 갑판 도장 |

- 정점 컬러 `COLOR_0`에 앰비언트 오클루전을 구워 넣었습니다. glTF 표준에 따라 baseColor에 곱해집니다.
  - three.js, Babylon.js, Godot에서는 자동으로 적용됩니다.
  - Unity나 Unreal의 기본 머티리얼은 정점 컬러를 무시할 수 있습니다. 이 경우 음영만 약간 평평해지고 나머지는 정상입니다.

## 회전·애니메이션용 노드

각 노드의 피벗은 실제 회전 중심에 있습니다. 노드를 Y축으로 돌리면 선회, 포신 노드를 X축으로 돌리면 앙각이 됩니다.

| 노드 | 설명 |
|---|---|
| `AK130_Turret` → `AK130_Guns` | 130 mm 연장포의 선회부와 포신 |
| `AK630_1` ~ `AK630_6` | 30 mm CIWS |
| `RBU6000_P`, `RBU6000_S` | 대잠 로켓 발사기 |
| `BassTilt_1` ~ `BassTilt_3`, `KiteScreech`, `PopGroup_P`, `PopGroup_S` | 사격통제 레이더 |
| `Fregat_Radar`, `TopPair_Radar`, `FrontDoor`, `TopDome` | 탐색·유도 레이더 |
| `OsaM_P`, `OsaM_S` | Osa-M 승강식 발사기 |
| `Crane` | 크레인 |
| `Propeller_P`, `Propeller_S` | 추진기 |

정적인 부분은 `Hull`, `Superstructure`, `Launchers`(P-500/P-1000 16기), `Weapons_VLS`(S-300F 8기), `Boats`로 묶여 있습니다.

## 고증 근거

- **선형과 배치**: A. Kuleshov의 1:100 도면(1993)을 사용했습니다. 측면도, 평면도, 선도(정면 선도), 상세도 5·6·7번을 서로 정합한 뒤 치수를 측정했습니다. 도면은 개인 참고용이라 저장소에 포함하지 않았습니다.
- **사진 교차 검증**: Moskva(2009·2012·2017년), Varyag(2011·2017년), Marshal Ustinov(1993·2018년)의 Wikimedia Commons 사진과 비교했습니다.
- **주요 치수**: 전장 186.4 m, 최대 폭 20.8 m, 흘수 6.28 m(소나 돔 포함 8.1 m)입니다. 건현은 함수 10.9 m, 중앙부 6.7 m, 후갑판 4.4 m입니다.
- **함미 갑판**: 함미 쪽 단차, 고정식 헬기갑판, VDS(가변심도 소나) 해치를 반영했습니다.

![profile](previews/slava_profile.png)

## 다른 함정 버전

`tools/slava_generator`로 함번과 함명을 바꿔 다시 생성할 수 있습니다.

```bash
cd tools/slava_generator
pip install -r requirements.txt
python build.py ../../assets/models/slava_class_cruiser.glb             # Moskva 121 (현재 파일)
python build.py varyag.glb  --ship varyag                                # Varyag 011
python build.py ustinov.glb --ship ustinov                               # Marshal Ustinov 055
```

## 크레디트

- 비교 이미지(`previews/slava_vs_ddg51.png`)에 나오는 DDG-51은 Yi Tsung Lee의 "1:1 Low poly US NAVY DDG-51 USS Arleigh Burke"이며, CC-BY-4.0 라이선스입니다.
- 함번과 함명 글꼴은 Big Shoulders(SIL OFL 1.1)와 DejaVu Sans Bold(Bitstream Vera 라이선스)를 썼습니다. 생성기 안에 라이선스 파일과 함께 들어 있습니다.
