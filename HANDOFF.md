# CarbideMatterGen — 세션 인수인계 (2026-06-01, rev2)

이 파일은 Claude Code 세션 전환용 인수인계 노트다. **새 세션은 이 폴더
(`/home/work/mattergen/CarbideMatterGen`)에서 시작**한다. 기존 위치
(`/home/work/CarbideMatterGen`)에서 NAS(20TB)로 폴더째 옮긴 직후 상태다.

> **rev2 핵심 변경 (이번 세션):** Stage B를 fairchem **v1 → v2(UMA)** 로 전환하고,
> 20개 SQS 전체 표면 ΔG_H* 스크린을 **완료**했다(4-way 병렬). 자세한 건 아래 3번 절.
> 메모리 `[[stage-b-fairchem-v2-uma]]`, `[[stage-b-cpu-bottleneck-single-worker]]` 참조.
>
> **Stage B 완료 ✅ (20/20, 오류 0).** 결과: `outputs/surface/dG_H_screen*.jsonl`
> + 요약 `outputs/surface/dG_H_summary.csv`. 그림 `figures/fig8_surface_screen.pdf`.
> 노트북 `notebooks/17_surface_screen.ipynb`(샤드 자동 병합), 논문 4장
> §sec:res-surf 결과 절 작성·컴파일 클린(29쪽).
>
> **핵심 결과:** 20종 전부 min|ΔG_H*|<0.011 eV(워크플로 검증). 변별은 활성점
> 밀도(2.3~15.3/nm², 6.6배). 최상위 = 란탄족 없는 d-블록 탄화물(#20 CrReMoIrRu).
> **엔트로피 역설**: S_mix↔밀도 r=−0.51(란탄족=희석). 프록시↔밀도 r=0.15(포화→
> 능동학습 근거). **다음 단계: Stage C(DFT 검증) 또는 능동학습 1회전.**

---

## 0. 폴더 이동 상태 (먼저 마무리할 것)

- 원본 `/home/work/CarbideMatterGen` → 사본 `/home/work/mattergen/CarbideMatterGen`
  로 `cp -a` 복사 완료. **검증됨**: 파일 20,373개 일치, git 최신 커밋 `7fb3dee`
  보존, GemNet-OC 체크포인트(155MB)·SQS CIF 20개·thesis.tex 모두 정상.
- **원본은 아직 삭제 안 함**(사용자 지시: 정상 가동 확인 전까지 유지).
- 이유: `/home/work`는 49GB 로컬 디스크(작음, 압박 심함), `/home/work/mattergen`은
  20TB NAS(여유 많음). 무거운 fairchem venv를 NAS에 둬야 디스크 문제 해결.
- **남은 일**: 새 세션에서 NAS 사본이 정상 동작(아래 1번) 확인 후,
  사용자 승인 받아 `rm -rf /home/work/CarbideMatterGen` 으로 원본 삭제.

### 새 세션 시작 방법 (사용자 안내)
사용자가 새 Claude Code 세션을 **작업 디렉토리
`/home/work/mattergen/CarbideMatterGen`** 에서 열면 된다. 그 세션에서 첫 지시로
이 `HANDOFF.md`를 읽게 하면 맥락이 복원된다. (예: "HANDOFF.md 읽고 이어서
진행해줘".)

---

## 1. 첫 확인 (새 세션이 제일 먼저 할 일)

```bash
# 패키지 import (PYTHONPATH = NAS 사본 경로)
PYTHONPATH=/home/work/mattergen/CarbideMatterGen \
  /home/work/OxideMatterGen/.venv/bin/python -c \
  "import carbidemattergen; from carbidemattergen import solid_solution, surface_screen; print('OK')"
# git 정상
git -C /home/work/mattergen/CarbideMatterGen log --oneline -3
# 디스크
df -h /home/work/mattergen | tail -1
```

주의: **명령을 `&&`로 길게 체이닝하지 말 것.** 이전 세션에서 긴 체이닝/
백그라운드+sleep 조합 때 출력 채널이 자주 끊겼다. 한 번에 짧은 명령 하나씩.

---

## 2. 프로젝트 현황 (무엇이 끝났나)

목표: 비백금족 HER 촉매 — 백금형 ΔG_H≈0, 금속성, 고엔트로피 탄화물.
워크플로: Stage A(생성/발견) → A'(SQS 고용체) → B(fairchem 표면 ΔG_H*) →
C(DFT) → 능동학습. **상세 배경은 메모리 `[[carbidemattergen-project]]` 참조**
(이전 세션이 누적 기록함).

완료된 것:
- **학습**: MatterGen 미세조정 600 epoch 완료. best ckpt
  `outputs/finetune_carbide/crystal-generation/mswvxapd/checkpoints/epoch=526-loss_val=0.14.ckpt`.
  (이 폴더 1.7GB — 정리하려면 best/last만 남기면 됨. 아직 안 함.)
- **생성**: 20,000개 (w=1.0 + w=2.5 각 1만). w-ablation 결과: w=2.5가 통과율
  8.2%→22.3%, ≥5금속 HEC 12→97 (8.1배) 우세. 전체 111 HEC.
  `outputs/gen/top_HEC_candidates.csv`.
- **Stage A' (SQS 고용체)**: `carbidemattergen/solid_solution.py` +
  `scripts/35_build_solid_solutions.py`. 상위 20 HEC → `outputs/sqs/*.cif`
  (20개, hcp-M2C 17 + rock-salt 3, 5~7금속, Φ 중앙값 159배 감소).
  최상위 SmGd3HoRe2GeMo3WC7 (7금속, S=1.82R).
- **논문**: `thesis/thesis.tex` 27쪽, xelatex 컴파일 클린. 서론+이론(확산모델)+
  방법+결과(fig4/5/6/7 + w-ablation 표) 통합. 참고문헌 12건 전부 DOI 검증.
  컴파일: `PATH=/home/work/texlive/bin/x86_64-linux:$PATH; latexmk -xelatex thesis.tex`
  (한글 kotex+xetexko, % !TEX program = xelatex).
- **노트북**: 10(loss), 15(generation stats, fig6), 16(SQS, fig7).

---

## 3. Stage B (fairchem 표면 ΔG_H*) — v2 전환 완료, 스크린 가동 중

SQS 고용체 표면에 H를 올려 실제 ΔG_H*를 계산(조성 프록시의 실측 검증). DFT 대신
fairchem **v2 UMA** 신경망으로 빠르게.

### 이번 세션에 한 일 (v1 → v2 전환)
- **v1은 폐기**: `fairchem-core==1.10.0`(OCPCalculator)은 torch_scatter/torch_sparse
  PyG 휠 필요 + torch 2.4.1+cpu/py3.12 휠이 깨짐(`_round_robin_process_groups`
  ImportError). 사용자 지시로 v2로 전환.
- **v2 확정**: `fairchem-core 2.20.0` + torch 2.8.0 (+pymatgen 별도 설치). 깨끗이
  import. API: `pretrained_mlip.get_predict_unit("uma-s-1p1", device="cuda")` →
  `FAIRChemCalculator(predictor, task_name="oc20")`. oc20 헤드=RPBE 총에너지라
  E(slab+H)/E(slab)/½E(H2) 일관 차감. UMA는 범용 모델이라 SQS의 란탄족/4d–5d
  이색 원소를 OC20 한정 GemNet-OC보다 잘 다룸(과학적으로도 우월).
- **환경**: `.venv_fc2` (NAS). (`.venv_fc1` 깨진 v1 잔재는 이번 세션에 삭제됨.)
- **인증**: UMA는 gated HF repo `facebook/UMA`. 사용자(jonglee69) 접근 승인됨,
  토큰은 `~/.cache/huggingface/token`. UMA GPU 메모리 ~1.3GB뿐.
- **GPU 안전성 실측**: A100 40GB 중 학습잡이 28GB·100% 점유. UMA 추가 시 peak
  1.3GB, 여유 11GB → **OOM 없음**. 단 **병목은 GPU가 아니라 4코어 CPU**(그래프
  구성). 학습잡이 ~2.7코어 사용 → 워커 1개만 띄움(초과구독 회피). `[[stage-b-cpu-bottleneck-single-worker]]`.
- **`surface_screen.py` 전면 재작성(v2 + 엄밀 프로토콜)**: Miller 지수별 최안정
  termination 선택(SQS는 대칭 없어 한 지수당 다수 종단 → 단일점 순위), 하부 절반
  고정 이완, 전 사이트 H 배치·이완, 면적당 열중립 밀도. **재개 가능**(이미 한
  후보는 skip). CLI: `--candidates --model --task --device --max-steps --site-cap`.

### 실행 중인 잡 (이어받을 때) — 4-way 병렬(샤딩)
OxideMatterGen 학습잡 종료로 GPU(38GB)·CPU(3코어) 여유 생김 → **4 워커 병렬**로
전환(후보를 `--shard i/4` 로 index%4 분배). 후보당 ~3h, 전체 **~15–18시간**.
```bash
pgrep -fa surface_screen | grep -c shard      # 워커 수(목표 4)
tail -f outputs/surface/screen.shard*.log     # 진행(후보당 [run]/[done])
cat outputs/surface/dG_H_screen*.jsonl | wc -l # 완료 후보 수 (목표 20)
```
- 결과는 `dG_H_screen.jsonl`(초기 01·02) + `dG_H_screen.shard{0..3}.jsonl` 에 분산.
  분석은 `dG_H_screen*.jsonl` 글롭으로 병합(노트북 17이 이미 그렇게 읽음).
- 재개: 어느 워커든 죽으면 그 샤드만 같은 명령 재실행(이미 끝난 후보는 모든
  `dG_H_screen*.jsonl` 을 읽어 skip). 4 워커 (재)기동:
```bash
cd /home/work/mattergen/CarbideMatterGen
for i in 0 1 2 3; do
  nohup env PYTHONPATH=$PWD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    .venv_fc2/bin/python -u -m carbidemattergen.surface_screen \
    --candidates outputs/sqs --model uma-s-1p1 --task oc20 --device cuda \
    --out outputs/surface/dG_H_screen.shard${i}.jsonl --max-steps 120 --shard ${i}/4 \
    > outputs/surface/screen.shard${i}.log 2>&1 &
done
```
(학습잡이 다시 돌면 4코어 초과구독되니 워커 수를 줄일 것. `[[stage-b-cpu-bottleneck-single-worker]]`.)

### 스크린 끝난 뒤 할 일
1. 노트북 `notebooks/17_surface_screen.ipynb` 재실행 → 표 + `figures/fig8_surface_screen.pdf`
   (부분 JSONL에도 동작하도록 작성됨).
2. `thesis/thesis.tex` 제4장에 **Stage B 결과 절** 신규 작성(현재 방법 절 §sec:m-surf만
   v2로 갱신됨; 결과 절은 실측 나온 뒤). 후보별 min|ΔG_H*|, 열중립 밀도, 프록시 vs
   실측 상관(fig8 패널 c).
3. **GemNet-OC ckpt 삭제됨**: v2 전환으로 미사용이라 이번 세션에 삭제(checkpoints/ 비어 있음).

### 주의 (이전 세션 교훈)
- 도구 호출은 짧게. 백그라운드+긴 sleep 지양(채널 끊김). 긴 잡은 nohup+모니터.
- 금속 정의: `_NONMETALS`는 hydrogen/extra_descriptors/solid_solution 모두
  동일해야 함(C/N/O/B/H/Si/할로겐만 비금속; Ge/Ga/Al/In/Sn은 금속 부격자).
  이미 통일됨 — 새 코드도 이 정의 따를 것.

---

## 3c. 경제성(비백금족) 재생성 — 진행 중

Stage B 결과(상위 후보 대부분이 PGM 포함; 청정 HEC는 풀에 1개뿐)를 받아, 사용자
지시로 **경제성 제약 조건화**를 추가해 재생성 중. 기존 20개는 비교군으로 고정.

**정책(사용자 확정):** PGM{Ru,Rh,Pd,Os,Ir,Pt} + 중희토류{Sm,Eu,Gd,Tb,Dy,Ho,Er,Tm,
Yb,Lu} = "비쌈". 비PGM d-블록 + 저가 경희토류{La,Ce,Pr,Nd,Y} + **Re 허용**.

**구현(커밋 `c4b76d3`):** 9번째 조건화 속성 `expensive_metal_fraction`
(=비싼 금속 분율). `hydrogen.py` 디스크립터 → `labeling.py`/`cache_overlay.py`
→ property YAML → `scripts/11_finetune_economic.sh`(9속성, 별도 출력
`outputs/finetune_economic`). 생성은 `scripts/21_generate_economic.sh`
(COND에 `expensive_metal_fraction:0.0`, 모델=경제성, 출력=`outputs/gen_economic`).

**⚠️ 리포 외부 편집:** 공유 mattergen 설치
`/home/work/OxideMatterGen/mattergen/mattergen/common/utils/globals.py`의
`PROPERTY_SOURCE_IDS`에 `"expensive_metal_fraction"` 추가함(필수). git 추적 안 됨.

**현황 (진행):**
- **재학습 완료 ✅**: `outputs/finetune_economic` best `epoch=371-loss_val=0.14`
  (8속성 베이스라인 0.14와 동일 품질). 학습셋 511구조, ~15분 소요. 손실 곡선
  노트북 `notebooks/12_loss_curve_economic.ipynb`(검증됨, fig1-4_*_economic).
- **조건화 검증 성공 ✅**: 파일럿 100개 생성 → **96% 청정(expensive=0), 0% PGM**
  (베이스라인 top-20은 90% PGM). 원소: Re·Mo·Ge·W·Cr + 저가 REE(Pr·Nd·Ce).
  청정 HEC 수율 9%. 재안내 생성이 의도대로 작동.
- **본 생성 가동 중**: `outputs/gen_economic_run.log`, w=2.5, ~2000개(10청크×200),
  `outputs/gen_economic/`. ~10h(GPU를 OxideMatterGen 생성잡과 공유). 재개 가능
  (.done 마커). pgrep -f mattergen-generate.

**다음:** 생성 완료 → `30_cheap_screen`로 청정 HEC 선별 → `35_build_solid_solutions`
SQS → `surface_screen`(UMA, `.venv_fc2`)로 활성점 밀도 → 기존 20개(baseline)와 비교.
이후 두 그룹에서 유망 후보 추려 DFT(Stage C).
**주의:** OxideMatterGen이 학습(28GB) 재개하면 OOM 위험 → batch/동시성 축소.

---

## 4. 파일 지도 (NAS 사본 기준 상대경로)

- `carbidemattergen/` — hydrogen(HER 프록시), features, extra_descriptors,
  labeling, cache_overlay, surface_screen(Stage B), solid_solution(Stage A').
- `scripts/` — 01 labels, 02 overlay, 10 finetune, 20 generate,
  30 cheap_screen, 35 build_solid_solutions, 40 active_learning, 50 dft.
- `outputs/` — finetune_carbide(학습), gen(생성 2만+screen csv+top_HEC),
  sqs(고용체 20 CIF + summary), **surface(Stage B: dG_H_screen.jsonl + screen.log)**.
- `.venv_fc2/` — **fairchem v2 환경**(2.20.0 + torch 2.8.0 + pymatgen). Stage B 실행용.
- `checkpoints/gemnet_oc_base_s2ef_all.pt` — fairchem v1 GemNet-OC. **v2 전환으로 미사용**(삭제 가능).
- `thesis/thesis.tex` — 통합 논문(정본). §sec:m-surf(표면 방법) v2로 갱신됨.
  introduction/methods.tex는 레거시.
- `notebooks/` — 10,15,16,**17(surface_screen, fig8)**.
- `figures/` — fig1-7 (gitignored, 노트북이 재생성). fig8=Stage B.

git 최신 커밋: `7fb3dee`. (HANDOFF.md, fairchem venv는 아직 미커밋 상태일 수
있음 — 새 세션에서 커밋.)
