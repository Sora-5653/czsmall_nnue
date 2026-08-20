# ADR 0002: 合法手生成ではrotation provenanceを失わない

## 状態

採用・実装方式は更新済み。

このADRが扱ったproject-side BFS generator自体は、後にpure Cobra movegenへ置換されました。一方で、**同一cellへ到達しても「最後の成功actionがrotationだったか」「どのkick/spin provenanceで到達したか」を失ってはいけない**という不変条件は現在も有効です。現行Cobra adapterはtargetのrotation/spin informationとcanonical pathを保持して同じ意味論を満たします。

## 背景

初期のlegal placement generatorでは、BFS keyとして `(x, y, rotation)` を使うのが自然に見えました。

しかしこれは不十分です。spin classificationは**どのようにその位置へ到達したか**に依存します。

- 最後のsuccessful actionがrotationである場合だけspinになり得る。
- T pieceでは、使用したkick indexがmini/full判定へ影響する。

したがって同じ `(x, y, rotation)` へ到達した2経路を、必ずしも同一stateとしてmergeできません。

旧実装でmergeしていたとき、emitted action setがBFS visit orderへ依存しました。これを発見したのはproperty testです。mirror-symmetricなSRS+ tableを使っているにもかかわらず、left/right mirrored boardで**legal placement数が一致しない**状態が175 sample中18 positionで起きました。

## 決定

旧BFS実装では、packed stateへ次を含めました。

- `arrived_by_rotation`
- kick index

これにより、同一座標でもrotationで到達したstateとslideで到達したstateを別nodeとして扱います。

現在は旧BFSを使用していませんが、設計上の決定を次の形で維持します。

> 合法配置をoutcome単位でmergeしても、spin判定に必要なrotation/spin provenanceと、そのplacementを再現するcanonical input sequenceを失ってはならない。

## 帰結

旧generatorでは次の効果がありました。

- mirror invarianceがcell単位で0 mismatchになった。
- state spaceは増えたが、packed keyは20 bitに収まり、ADR 0003のflat visited tableを可能にした。
- BFS visit orderの偶然によってspin-bearing placementが消えることを防いだ。

現行Cobra implementationでは、旧20-bit BFS stateそのものは存在しません。そのためこのADRの具体的data structureはhistorical recordであり、現在のcode requirementは**provenance preservation**です。
