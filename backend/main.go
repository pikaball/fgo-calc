package main

import (
	"container/heap"
	"encoding/json"
	"net/http"
	"os"
	"sort"
	"strconv"
	"sync"
	"runtime"
	"time"
	"log"

	"github.com/gin-gonic/gin"
)

const OPTIMIZE_LIMIT = 5 // 只输出最优的5个组合

type ServantDetail struct {
	Name     string            `json:"name"`
	Traits   []int             `json:"traits"`
	Cost     int               `json:"cost"`
	Img      string            `json:"img"`
	TraitSet map[int]struct{}  `json:"-"`
}

type Servant struct {
	Id   int                       `json:"id"`
	Name string                    `json:"name"`
	Diff map[string]ServantDetail `json:"diff"`
}

type Filter struct {
	Traits []int
	Effect float64
}

type CraftEssence struct {
	Id      int      `json:"id"`
	Name    string   `json:"name"`
	Img     string   `json:"img"`
	Cost    int      `json:"cost"`
	Filters []Filter `json:"filters"`
}

type SvtBonus struct {
	Svt     *Servant
	DiffKey string
	Bonus   int
	Cost    int
}

type Team struct {
	Servants      []Servant
	DiffChoice    []string
	CraftEssences []CraftEssence
	TotalCost     int
	TotalBond     int
}

// TeamHeap 实现 heap.Interface，用于维护 Top K
// 这是一个最小堆，堆顶存储的是 Top K 中“最差”的那个方案
type TeamHeap []Team

func (h TeamHeap) Len() int { return len(h) }
func (h TeamHeap) Less(i, j int) bool {
    // 比较逻辑：Bond 小的在前；Bond 相等 Cost 小的在前
    // 这样堆顶就是 Bond 最低（或 Cost 最低）的元素
    if h[i].TotalBond != h[j].TotalBond {
        return h[i].TotalBond < h[j].TotalBond
    }
    return h[i].TotalCost < h[j].TotalCost
}
func (h TeamHeap) Swap(i, j int) { h[i], h[j] = h[j], h[i] }
func (h *TeamHeap) Push(x interface{}) {
    *h = append(*h, x.(Team))
}
func (h *TeamHeap) Pop() interface{} {
    old := *h
    n := len(old)
    x := old[n-1]
    *h = old[0 : n-1]
    return x
}

type TeamResultCE struct {
    CraftEssence
    Contribution int `json:"contribution"`
}

type TeamResponse struct {
    Servants      []Servant      `json:"Servants"`
    DiffChoice    []string       `json:"DiffChoice"`
    CraftEssences []TeamResultCE `json:"CraftEssences"`
    TotalCost     int            `json:"TotalCost"`
    TotalBond     int            `json:"TotalBond"`
}

type PathNode struct {
    ItemIdx int
    Prev    *PathNode
}

type CeEffect struct {
	Percent float64
	Direct  int
}

type DPResult struct {
	Bond  int
	Combo []SvtBonus
}

var (
	ceEffects       map[int]map[int]map[string]CeEffect // ceId -> svtId -> diffKey -> effect
	ceEffectsOnce   sync.Once
	servants        []Servant
	craftEssences   []CraftEssence
	traits          map[int]string
	dominateMap     map[int]int
)

func (f *Filter) UnmarshalJSON(data []byte) error {
	var raw []json.RawMessage
	json.Unmarshal(data, &raw)
	json.Unmarshal(raw[0], &f.Traits)
	json.Unmarshal(raw[1], &f.Effect)
	return nil
}

func LoadInfo() {
	svtdata, _ := os.ReadFile("../data/servants.json")
	json.Unmarshal(svtdata, &servants)

	for i := range servants {
		for key, detail := range servants[i].Diff {
			traitSet := make(map[int]struct{}, len(detail.Traits))
			for _, traitId := range detail.Traits {
				traitSet[traitId] = struct{}{}
			}
			detail.TraitSet = traitSet
			servants[i].Diff[key] = detail
		}
	}

	cedata, _ := os.ReadFile("../data/ces.json")
	json.Unmarshal(cedata, &craftEssences)

	traitmap, _ := os.ReadFile("../data/names/traits.json")
	json.Unmarshal(traitmap, &traits)

	PrecomputeCeEffects()
	BuildDominateMap()
}

// ========== PrecomputeCeEffects: 预计算每张CE对每个从者形态的效果 ==========
func PrecomputeCeEffects() {
	ceEffects = make(map[int]map[int]map[string]CeEffect)
	for _, ce := range craftEssences {
		ceEffects[ce.Id] = make(map[int]map[string]CeEffect)
		for _, svt := range servants {
			ceEffects[ce.Id][svt.Id] = make(map[string]CeEffect)
			for diffKey, detail := range svt.Diff {
				percent := 0.0
				direct := 0
				for _, filter := range ce.Filters {
					match := true
					if len(filter.Traits) > 0 {
						for _, tr := range filter.Traits {
							if _, ok := detail.TraitSet[tr]; !ok {
								match = false
								break
							}
						}
					}
					if match {
						if filter.Effect > 0 {
							percent += filter.Effect
						} else {
							direct += int(-filter.Effect)
						}
						break
					}
				}
				ceEffects[ce.Id][svt.Id][diffKey] = CeEffect{Percent: percent, Direct: direct}
			}
		}
	}
	ceEffectsOnce.Do(func() {})
}

// ========== computeEffectsForCombo: 用预计算表快速合并 CE 组合效果 ==========
func computeEffectsForCombo(ceCombo []CraftEssence, svtId int, diffKey string) (float64, int) {
	totalPercent := 0.0
	totalDirect := 0
	for _, ce := range ceCombo {
		if m1, ok := ceEffects[ce.Id]; ok {
			if m2, ok2 := m1[svtId]; ok2 {
				if eff, ok3 := m2[diffKey]; ok3 {
					totalPercent += eff.Percent
					totalDirect += eff.Direct
				}
			}
		}
	}
	return totalPercent, totalDirect
}

// 构建五星通用无特性礼装的“绝对优于”关系
func BuildDominateMap() {
    nonFilterCe := []CraftEssence{}
    for _, ce := range craftEssences {
        if len(ce.Filters) == 1 && ce.Cost == 12 {
            if len(ce.Filters[0].Traits) == 0 {
                nonFilterCe = append(nonFilterCe, ce)
            }
        }
    }
    sort.Slice(nonFilterCe, func(i, j int) bool {
        // 增加 ID 作为 Tie-breaker，确保排序确定性
        if nonFilterCe[i].Filters[0].Effect != nonFilterCe[j].Filters[0].Effect {
            return nonFilterCe[i].Filters[0].Effect > nonFilterCe[j].Filters[0].Effect
        }
        return nonFilterCe[i].Id < nonFilterCe[j].Id
    })
    dominateMap = make(map[int]int)
    for i := 0; i < len(nonFilterCe)-1; i++ {
        dominateMap[nonFilterCe[i+1].Id] = nonFilterCe[i].Id
    }
}

func FindInPool(id int, cePool []CraftEssence, included []CraftEssence) bool {
	for _, ce := range cePool {
		if ce.Id == id {
			return true
		}
	}
	for _, ce := range included {
		if ce.Id == id {
			return true
		}
	}
	return false
}

func FixDominateMap(cePool []CraftEssence, included []CraftEssence) map[int]int {
	fixedMap := make(map[int]int)
	for B, A := range dominateMap {
		if !FindInPool(B, cePool, included) {
			continue
		}
		currentA := A
		for {
			if !FindInPool(currentA, cePool, included) {
				if nextA, ok := dominateMap[currentA]; ok {
					currentA = nextA
				} else {
					currentA = -1
					break
				}
			} else {
				break
			}
		}
		if currentA != -1 {
			fixedMap[B] = currentA
		}
	}
	return fixedMap
}

func GetCombination(num int, includeCe []int, excludeCe []int) [][]CraftEssence {
	if num <= 0 {
		return [][]CraftEssence{}
	}
	includeSet := map[int]bool{}
	excludeSet := map[int]bool{}
	for _, id := range includeCe {
		includeSet[id] = true
	}
	for _, id := range excludeCe {
		excludeSet[id] = true
	}

	included := []CraftEssence{}
	pool := []CraftEssence{}
	for _, ce := range craftEssences {
		if excludeSet[ce.Id] {
			continue
		}
		if includeSet[ce.Id] {
			included = append(included, ce)
		} else {
			pool = append(pool, ce)
		}
	}

	if len(pool) < num {
		comb := make([]CraftEssence, len(pool))
		copy(comb, pool)
		return [][]CraftEssence{comb}
	}

	if len(included) > num {
		return [][]CraftEssence{}
	}
	need := num - len(included)
	if need == 0 {
		comb := make([]CraftEssence, len(included))
		copy(comb, included)
		return [][]CraftEssence{comb}
	}

	// 对 pool 进行排序，确保上位礼装在下位礼装之前，排序逻辑与 BuildDominateMap 保持一致
    sort.Slice(pool, func(i, j int) bool {
        eff1 := 0.0
        if len(pool[i].Filters) > 0 {
            eff1 = pool[i].Filters[0].Effect
        }
        eff2 := 0.0
        if len(pool[j].Filters) > 0 {
            eff2 = pool[j].Filters[0].Effect
        }
        if eff1 != eff2 {
            return eff1 > eff2
        }
        return pool[i].Id < pool[j].Id
    })

	results := [][]CraftEssence{}
	var dfs func(start int, picked []CraftEssence, pickedSet map[int]bool)
	fixedDominateMap := FixDominateMap(pool, included)
	initialPickedSet := make(map[int]bool)
    for _, ce := range included {
        initialPickedSet[ce.Id] = true
    }
	dfs = func(start int, picked []CraftEssence, pickedSet map[int]bool) {
		if len(picked) == need {
			comb := make([]CraftEssence, 0, num)
			comb = append(comb, included...)
			comb = append(comb, picked...)
			results = append(results, comb)
			return
		}
		remainSlots := need - len(picked)
		for i := start; i <= len(pool)-remainSlots; i++ {
			ce := pool[i]
			if domA, ok := fixedDominateMap[ce.Id]; ok {
				if !pickedSet[domA] {
					continue
				}
			}
			pickedSet[ce.Id] = true
			dfs(i+1, append(picked, pool[i]), pickedSet)
			delete(pickedSet, ce.Id)
		}
	}
	dfs(0, []CraftEssence{}, initialPickedSet)
	return results
}

func FilterServants(traits []int, includeSvt []int, excludeSvt []int) []Servant {
	includeSet := map[int]bool{}
	excludeSet := map[int]bool{}
	for _, id := range includeSvt {
		includeSet[id] = true
	}
	for _, id := range excludeSvt {
		excludeSet[id] = true
	}

	result := []Servant{}
	traitSet := map[int]bool{}
    for _, t := range traits {
        traitSet[t] = true
    }

    for _, svt := range servants {
        if excludeSet[svt.Id] {
            continue
        }
        if includeSet[svt.Id] {
            result = append(result, svt)
            continue
        }
        if len(traits) == 0 {
            result = append(result, svt)
            continue
        }

        matched := false
        for _, detail := range svt.Diff {
            for _, st := range detail.Traits {
                if traitSet[st] {
                    matched = true
                    break
                }
            }
            if matched {
                break
            }
        }
        if matched {
            result = append(result, svt)
        }
    }
    return result
}

// ---------- 优化关键：使用状态池的轻量 DP 实现 ----------

// DPState: 存储最小信息以便回溯
type DPState struct {
	Bond     int
	Prev     int // index in statePool of previous state, -1 for none
	SvtIndex int // index into optionalBonuses; -1 for base
}

// insertTopKIndices: 给 dp slot（存 state indices）插入 newIdx，按 Bond 降序，保持 <= k 个
func insertTopKIndices(slot []int, statePool []DPState, newIdx int, k int) []int {
	if len(slot) == 0 {
		return []int{newIdx}
	}
	newBond := statePool[newIdx].Bond
	res := make([]int, 0, min(len(slot)+1, k))
	inserted := false
	for i := 0; i < len(slot); i++ {
		if !inserted && newBond > statePool[slot[i]].Bond {
			res = append(res, newIdx)
			inserted = true
			if len(res) >= k {
				break
			}
		}
		res = append(res, slot[i])
		if len(res) >= k {
			break
		}
	}
	if !inserted && len(res) < k {
		res = append(res, newIdx)
	}
	return res
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func Optimize(costLimit int, svtLimit int, ceLimit int, allowTraits []int, includeSvt []int, includeSvtDiff []string, excludeSvt []int, includeCe []int, excludeCe []int, baseBond int) ([]TeamResponse, time.Duration) {
	startTime := time.Now()
	log.Println("Optimize called with costLimit:", costLimit, "svtLimit:", svtLimit, "ceLimit:", ceLimit)
	if len(includeSvt) > svtLimit {
		return []TeamResponse{}, 0
	}
	if len(includeCe) > ceLimit {
		return []TeamResponse{}, 0
	}

	mince := len(includeCe)
	if mince < 0 {
		mince = 0
	}
	mince += (costLimit - svtLimit*16) / 12

	if mince > ceLimit {
		mince = ceLimit
	}

	cePool := [][]CraftEssence{}
	for i := mince; i <= ceLimit; i++ {
		combs := GetCombination(i, includeCe, excludeCe)
		if len(combs) == 1 {
			comb := combs[0]
			if len(comb) < i {
				cePool = append(cePool, comb)
				break
			}
		}
		cePool = append(cePool, combs...)
	}

	log.Println("CE Pool: ", len(cePool))

	// 过滤从者池（一次性）
	svtPool := FilterServants(allowTraits, includeSvt, excludeSvt)

	log.Println("Servant Pool: ", len(svtPool))

	// 预构造 includeSvtSet，避免在 worker 内重复构造
	includeSvtSet := map[int]bool{}
    for _, id := range includeSvt {
        includeSvtSet[id] = true
    }
    includeSvtDiffMap := make(map[int]string)
    for i, id := range includeSvt {
        if i < len(includeSvtDiff) {
            includeSvtDiffMap[id] = includeSvtDiff[i]
        }
    }

	// 并发 worker 处理 CE 组合（保持并发模型）
	numWorkers := runtime.GOMAXPROCS(0)
	ceJobs := make(chan []CraftEssence, len(cePool))
	resultsChan := make(chan []Team, len(cePool))
	var wg sync.WaitGroup

	worker := func() {
		defer wg.Done()
		for ceCombo := range ceJobs {
			localTeams := []Team{}

			// CE 总 cost
			ceCost := 0
			for _, ce := range ceCombo {
				ceCost += ce.Cost
			}
			// 早期剪枝：CE cost 已超过全局 costLimit 时跳过（必须保留这个基础的剪枝）
			if ceCost > costLimit {
				resultsChan <- localTeams
				continue
			}

			// 1) 处理必选从者：为每个必选从者选出在此 ceCombo 下的最佳形态
			mandatoryBonuses := []SvtBonus{}
            currentSvtPool := make([]Servant, 0, len(svtPool))
            for i := range svtPool {
                svt := &svtPool[i]
                if includeSvtSet[svt.Id] {
                    // 如果指定了形态，则直接使用
                    if diffKey, ok := includeSvtDiffMap[svt.Id]; ok {
                        if detail, ok := svt.Diff[diffKey]; ok {
                            totalPercent, totalDirect := computeEffectsForCombo(ceCombo, svt.Id, diffKey)
                            bonus := int(float64(baseBond)*totalPercent/100.0) + totalDirect + baseBond
                            mandatoryBonuses = append(mandatoryBonuses, SvtBonus{
                                Svt:     svt,
                                DiffKey: diffKey,
                                Bonus:   bonus,
                                Cost:    detail.Cost,
                            })
                            continue // 处理完这个必选从者
                        }
                    }
                    // 未指定形态或指定形态无效，则计算最佳形态
                    totalPercent, totalDirect := computeEffectsForCombo(ceCombo, svt.Id, "default")
					maxBonus := int(float64(baseBond)*totalPercent/100.0) + totalDirect + baseBond
					bestDiffKey := "default"
					bestCost := svt.Diff["default"].Cost
					for key, detail := range svt.Diff {
						totalPercent, totalDirect := computeEffectsForCombo(ceCombo, svt.Id, key)
						currentBonus := int(float64(baseBond)*totalPercent/100.0) + totalDirect + baseBond
						if currentBonus > maxBonus {
							maxBonus = currentBonus
							bestDiffKey = key
							bestCost = detail.Cost
						}
					}
					if bestDiffKey != "" {
						mandatoryBonuses = append(mandatoryBonuses, SvtBonus{
							Svt:     svt,
							DiffKey: bestDiffKey,
							Bonus:   maxBonus,
							Cost:    bestCost,
						})
					}
				} else {
					currentSvtPool = append(currentSvtPool, *svt)
				}
			}

			mandatoryCost := 0
			mandatoryBond := 0
			for _, mb := range mandatoryBonuses {
				mandatoryCost += mb.Cost
				mandatoryBond += mb.Bonus
			}

			currentCostLimit := costLimit - ceCost - mandatoryCost
			currentSvtLimit := svtLimit - len(mandatoryBonuses)
			if currentCostLimit < 0 || currentSvtLimit < 0 {
				// 无法满足 cost 或 slot
				resultsChan <- localTeams
				continue
			}

			// 如果不需要选可选从者（只有必选）
			if currentSvtLimit == 0 {
				team := Team{
					CraftEssences: ceCombo,
					TotalBond:     mandatoryBond,
					TotalCost:     ceCost + mandatoryCost,
				}
				for _, sb := range mandatoryBonuses {
					team.Servants = append(team.Servants, *sb.Svt)
					team.DiffChoice = append(team.DiffChoice, sb.DiffKey)
				}
				localTeams = append(localTeams, team)
				resultsChan <- localTeams
				continue
			}

			// 2) 对可选从者，为每个从者选一个最佳形态（显著减小 DP 输入）
			optionalBonuses := make([]SvtBonus, 0, len(currentSvtPool))
			for i := range currentSvtPool {
				svt := &currentSvtPool[i]
				// 先固定默认形态
				totalPercent, totalDirect := computeEffectsForCombo(ceCombo, svt.Id, "default")
				maxBonus := int(float64(baseBond)*totalPercent/100.0) + totalDirect + baseBond
				bestDiffKey := "default"
				bestCost := svt.Diff["default"].Cost
				for key, detail := range svt.Diff {
					if key == "default" {
						continue
					}
					totalPercent, totalDirect := computeEffectsForCombo(ceCombo, svt.Id, key)
					currentBonus := int(float64(baseBond)*totalPercent/100.0) + totalDirect + baseBond
					if currentBonus > maxBonus {
						maxBonus = currentBonus
						bestDiffKey = key
						bestCost = detail.Cost
					}
				}
				if bestDiffKey != "" {
					optionalBonuses = append(optionalBonuses, SvtBonus{
						Svt:     svt,
						DiffKey: bestDiffKey,
						Bonus:   maxBonus,
						Cost:    bestCost,
					})
				}
			}

			// 如果没有可选项
			if len(optionalBonuses) == 0 {
				team := Team{
					CraftEssences: ceCombo,
					TotalBond:     mandatoryBond,
					TotalCost:     ceCost + mandatoryCost,
				}
				for _, sb := range mandatoryBonuses {
					team.Servants = append(team.Servants, *sb.Svt)
					team.DiffChoice = append(team.DiffChoice, sb.DiffKey)
				}
				localTeams = append(localTeams, team)
				resultsChan <- localTeams
				continue
			}

			// 3) DP：使用 dp[k][c] = best bond（k = chosen count），并保存回溯信息
			// dp 大小固定且小： (0..currentSvtLimit) x (0..currentCostLimit)
			// 初始化 dp 为 -inf（不可能）
			const NEG = -1 << 60
			dp := make([][]int, currentSvtLimit+1)
			for i := range dp {
				dp[i] = make([]int, currentCostLimit+1)
				for j := range dp[i] {
					dp[i][j] = NEG
				}
			}
			// 0 个选取，0 cost => bond 0
			dp[0][0] = 0

			// 为回溯保存信息
			// prevChoice[k][c] = index of item used to reach dp[k][c], -1 if none
			// prevCost[k][c] = previous cost before using that item
			// prevChoice := make([][]int, currentSvtLimit+1)
			// prevCost := make([][]int, currentSvtLimit+1)
			// for i := range prevChoice {
			// 	prevChoice[i] = make([]int, currentCostLimit+1)
			// 	prevCost[i] = make([]int, currentCostLimit+1)
			// 	for j := range prevChoice[i] {
			// 		prevChoice[i][j] = -1
			// 		prevCost[i][j] = -1
			// 	}
			// }

			paths := make([][]*PathNode, currentSvtLimit+1)
            for i := range paths {
                paths[i] = make([]*PathNode, currentCostLimit+1)
            }

			// 遍历每个 item（每个可选从者），做 0/1 背包（并且项数受限）
			for itemIdx, item := range optionalBonuses {
				cost := item.Cost
				bonus := item.Bonus
				// 如果单项成本超过限制，跳过
				if cost > currentCostLimit {
					continue
				}
				// k 从 high 到 low，cost 从 high 到 low（0/1 背包）
				for k := currentSvtLimit; k >= 1; k-- {
					// j: cost
					// j from currentCostLimit down to cost
					for j := currentCostLimit; j >= cost; j-- {
						if dp[k-1][j-cost] == NEG {
							continue
						}
						newBond := dp[k-1][j-cost] + bonus
						if newBond > dp[k][j] {
                            dp[k][j] = newBond
                            paths[k][j] = &PathNode{
                                ItemIdx: itemIdx,
                                Prev:    paths[k-1][j-cost],
                            }
                        }
					}
				}
			}

			// 4) 收集结果：遍历所有 k (1..currentSvtLimit) 与 cost，并回溯
			for k := 1; k <= currentSvtLimit; k++ {
				for j := 0; j <= currentCostLimit; j++ {
					if dp[k][j] == NEG {
						continue
					}
					// 回溯出选中的 items
					used := make([]bool, len(optionalBonuses))
					node := paths[k][j]
                    for node != nil {
                        used[node.ItemIdx] = true
                        node = node.Prev
                    }
					// 根据 used 标记构造 chosen 列表和计算总成本
					chosen := []SvtBonus{}
					totalCost := ceCost + mandatoryCost
					for idx, flag := range used {
						if flag {
							sb := optionalBonuses[idx]
							chosen = append(chosen, sb)
							totalCost += sb.Cost
						}
					}

					team := Team{
						CraftEssences: ceCombo,
						TotalBond:     mandatoryBond,
					}
					// 添加必选从者
					for _, sb := range mandatoryBonuses {
						team.Servants = append(team.Servants, *sb.Svt)
						team.DiffChoice = append(team.DiffChoice, sb.DiffKey)
					}
					// 添加 DP 选出的从者
					for _, sb := range chosen {
						team.Servants = append(team.Servants, *sb.Svt)
						team.DiffChoice = append(team.DiffChoice, sb.DiffKey)
						team.TotalBond += sb.Bonus
					}
					team.TotalCost = totalCost
					localTeams = append(localTeams, team)
				}
			}

			resultsChan <- localTeams
		}
	}

	// 启动 worker
	wg.Add(numWorkers)
	for i := 0; i < numWorkers; i++ {
		go worker()
	}

	// 发任务
	go func() {
		for _, ceCombo := range cePool {
			ceJobs <- ceCombo
		}
		close(ceJobs)
	}()

	// 等待并收集
	go func() {
		wg.Wait()
		close(resultsChan)
	}()

	// 使用固定大小的堆来收集结果，避免内存爆炸
    h := &TeamHeap{}
    heap.Init(h)

    for teams := range resultsChan {
        for _, team := range teams {
            if h.Len() < OPTIMIZE_LIMIT {
                heap.Push(h, team)
            } else {
                // 如果当前 team 比堆顶（Top K 中最差的）要好，则替换
                // "好" 的定义是：Bond 更高，或者 Bond 相等但 Cost 更高
                top := (*h)[0]
                if team.TotalBond > top.TotalBond || (team.TotalBond == top.TotalBond && team.TotalCost > top.TotalCost) {
                    (*h)[0] = team
                    heap.Fix(h, 0)
                }
            }
        }
    }

	log.Println("Dp Done. ")

	// 堆中元素弹出是从小到大（最差到最好），我们需要最好到最差
    // 所以先弹出到 slice，然后反转（或者倒序填充）
    limit := h.Len()
    sortedTeams := make([]Team, limit)
    for i := limit - 1; i >= 0; i-- {
        sortedTeams[i] = heap.Pop(h).(Team)
    }

	finalResults := make([]TeamResponse, 0, limit)

	// if len(allTeams) > 0 {
	// 	uniqueTeams := []Team{allTeams[0]}
	// 	for i := 1; i < len(allTeams); i++ {
	// 		isDup := false
	// 		for _, ut := range uniqueTeams {
	// 			if ut.TotalBond == allTeams[i].TotalBond && ut.TotalCost == allTeams[i].TotalCost {
	// 				isDup = true
	// 				break
	// 			}
	// 		}
	// 		if !isDup {
	// 			uniqueTeams = append(uniqueTeams, allTeams[i])
	// 			if len(uniqueTeams) >= OPTIMIZE_LIMIT {
	// 				break
	// 			}
	// 		}
	// 	}
	// 	allTeams = uniqueTeams
	// }

	for i := 0; i < limit; i++ {
		team := sortedTeams[i]
		response := TeamResponse{
			Servants:      team.Servants,
			DiffChoice:    team.DiffChoice,
			TotalCost:     team.TotalCost,
			TotalBond:     team.TotalBond,
			CraftEssences: make([]TeamResultCE, len(team.CraftEssences)),
		}

		for j, ce := range team.CraftEssences {
			totalContribution := 0
			for k, svt := range team.Servants {
				diffKey := team.DiffChoice[k]
				if m1, ok := ceEffects[ce.Id]; ok {
					if m2, ok2 := m1[svt.Id]; ok2 {
						if eff, ok3 := m2[diffKey]; ok3 {
							totalContribution += int(float64(baseBond)*eff.Percent/100.0) + eff.Direct
						}
					}
				}
			}
			response.CraftEssences[j] = TeamResultCE{
				CraftEssence: ce,
				Contribution: totalContribution,
			}
		}
		finalResults = append(finalResults, response)
	}

	log.Println("Optimization complete. Total time:", time.Since(startTime))

	return finalResults, time.Since(startTime)
}


func MapStr2Int(data []string) []int {
	result := []int{}
	for _, d := range data {
		if v, err := strconv.Atoi(d); err == nil {
			result = append(result, v)
		}
	}
	return result
}

func main() {
	r := gin.Default()

	r.NoRoute(func(c *gin.Context) {
		c.File("./static/index.html")
	})

	r.Static("/static", "./static")

	r.GET("/api/data", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{
			"servants":      servants,
			"craftEssences": craftEssences,
			"traits":        traits,
		})
	})

	r.GET("/test", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{"msg": "给我玩FGO"})
	})

	r.POST("/api/filtertraits", func(c *gin.Context) {
		c.JSON(http.StatusOK, FilterServants(MapStr2Int(c.PostFormArray("traits")), []int{}, []int{}))
	})
	r.POST("/api/calculate", func(c *gin.Context) {
		costLimit, _ := strconv.Atoi(c.PostForm("costlimit"))
		svtLimit, _ := strconv.Atoi(c.PostForm("svtlimit"))
		ceLimit, _ := strconv.Atoi(c.PostForm("celimit"))
		if ceLimit > 6 {
			c.JSON(http.StatusBadRequest, gin.H{"error": "礼装数量不能超过6个"})
			return
		}
		if svtLimit > 6 {
			c.JSON(http.StatusBadRequest, gin.H{"error": "从者数量不能超过6个"})
			return
		}
		if costLimit > 130 {
			c.JSON(http.StatusBadRequest, gin.H{"error": "总cost不能超过130"})
			return
		}
		baseBond, _ := strconv.Atoi(c.PostForm("basebond"))
        results, duration := Optimize(costLimit,
            svtLimit,
            ceLimit,
            MapStr2Int(c.PostFormArray("allowtraits")),
            MapStr2Int(c.PostFormArray("includesvt")),
            c.PostFormArray("includesvtdiff"),
            MapStr2Int(c.PostFormArray("excludesvt")),
            MapStr2Int(c.PostFormArray("includece")),
            MapStr2Int(c.PostFormArray("excludece")),
            baseBond)
		c.JSON(http.StatusOK, gin.H{
			"teams":  results,
			"duration": duration.Milliseconds(),
		})
	})
	LoadInfo()

	r.Run(":30005")
}
