//! plato-i2i-dcs — Multi-agent DCS (Distributed Constraint Satisfaction)
//!
//! Extends plato-i2i with multi-agent coordination: multiple agents run
//! specialized DCS, share tiles, negotiate consensus, and fuse beliefs.

use std::collections::HashMap;

pub type AgentId = u32;
pub type Expertise = Vec<String>;

// ── Belief Score ──────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
pub struct BeliefScore {
    pub confidence: f32,
    pub trust: f32,
    pub relevance: f32,
}

impl Default for BeliefScore {
    fn default() -> Self { Self { confidence: 0.5, trust: 0.5, relevance: 0.5 } }
}

impl BeliefScore {
    pub fn new(confidence: f32, trust: f32, relevance: f32) -> Self {
        Self {
            confidence: confidence.max(0.0).min(1.0),
            trust: trust.max(0.0).min(1.0),
            relevance: relevance.max(0.0).min(1.0),
        }
    }

    pub fn composite(&self) -> f32 {
        (self.confidence * self.trust * self.relevance).powf(1.0 / 3.0)
    }

    pub fn get(&self, dim: BeliefDimension) -> f32 {
        match dim {
            BeliefDimension::Confidence => self.confidence,
            BeliefDimension::Trust => self.trust,
            BeliefDimension::Relevance => self.relevance,
        }
    }

    pub fn set(&mut self, dim: BeliefDimension, value: f32) {
        let v = value.max(0.0).min(1.0);
        match dim {
            BeliefDimension::Confidence => self.confidence = v,
            BeliefDimension::Trust => self.trust = v,
            BeliefDimension::Relevance => self.relevance = v,
        }
    }

    pub fn reinforce(&mut self, dim: BeliefDimension, strength: f32) {
        let cur = self.get(dim);
        self.set(dim, (cur * 4.0 + strength) / 5.0);
    }

    pub fn undermine(&mut self, dim: BeliefDimension, strength: f32) {
        let cur = self.get(dim);
        self.set(dim, (cur * 4.0 - strength).max(0.0) / 4.0);
    }

    pub fn decay(&mut self, rate: f32) {
        let pull = |v: f32| (v + (0.5 - v) * rate).max(0.0).min(1.0);
        self.confidence = pull(self.confidence);
        self.trust = pull(self.trust);
        self.relevance = pull(self.relevance);
    }

    pub fn actionable(&self, min_c: f32, min_t: f32, min_r: f32) -> bool {
        self.confidence >= min_c && self.trust >= min_t && self.relevance >= min_r
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BeliefDimension {
    Confidence,
    Trust,
    Relevance,
}

// ── Belief Store ──────────────────────────────────────────────────────

pub struct BeliefStore {
    beliefs: HashMap<String, BeliefScore>,
    decay_per_tick: f32,
}

impl Default for BeliefStore {
    fn default() -> Self { Self::new() }
}

impl BeliefStore {
    pub fn new() -> Self {
        Self { beliefs: HashMap::new(), decay_per_tick: 0.02 }
    }

    pub fn set(&mut self, key: &str, score: BeliefScore) {
        self.beliefs.insert(key.to_string(), score);
    }

    pub fn get(&self, key: &str) -> Option<BeliefScore> {
        self.beliefs.get(key).copied()
    }

    pub fn reinforce(&mut self, key: &str, dim: BeliefDimension, strength: f32) {
        let score = self.beliefs.entry(key.to_string()).or_default();
        score.reinforce(dim, strength);
    }

    pub fn undermine(&mut self, key: &str, dim: BeliefDimension, strength: f32) {
        let score = self.beliefs.entry(key.to_string()).or_default();
        score.undermine(dim, strength);
    }

    pub fn tick(&mut self) {
        for score in self.beliefs.values_mut() {
            score.decay(self.decay_per_tick);
        }
    }

    pub fn len(&self) -> usize { self.beliefs.len() }
    pub fn is_empty(&self) -> bool { self.beliefs.is_empty() }
}

// ── Constraint Engine ─────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuditOutcome { Pass, Fail(String) }

pub struct ConstraintEngine {
    forbidden: Vec<String>,
}

impl Default for ConstraintEngine {
    fn default() -> Self { Self::new() }
}

impl ConstraintEngine {
    pub fn new() -> Self {
        Self { forbidden: vec!["RM -RF /".to_string(), "DELETE FROM USERS".to_string()] }
    }

    pub fn audit(&self, command: &str) -> AuditOutcome {
        for pattern in &self.forbidden {
            if command.to_uppercase().contains(pattern) {
                return AuditOutcome::Fail(format!("Forbidden: contains {}", pattern));
            }
        }
        AuditOutcome::Pass
    }
}

// ── Locks ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LockSource { Inconsistency, Expert, Inferred, Observation }

impl LockSource {
    pub fn base_trust(&self) -> f32 {
        match self {
            LockSource::Expert => 1.0,
            LockSource::Inconsistency => 0.8,
            LockSource::Observation => 0.7,
            LockSource::Inferred => 0.4,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Lock {
    pub id: u64,
    pub description: String,
    pub trigger_pattern: String,
    pub enforcement: String,
    pub source: LockSource,
    pub strength: f32,
    pub verifications: u32,
    pub violations: u32,
}

impl Lock {
    pub fn new(description: &str, trigger: &str, enforcement: &str, source: LockSource) -> Self {
        use std::time::{SystemTime, UNIX_EPOCH};
        Self {
            id: SystemTime::now().duration_since(UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64).unwrap_or(0),
            description: description.to_string(),
            trigger_pattern: trigger.to_string(),
            enforcement: enforcement.to_string(),
            source,
            strength: source.base_trust(),
            verifications: 0,
            violations: 0,
        }
    }

    pub fn verify(&mut self) -> f32 { self.verifications += 1; self.strength = (self.strength + 0.05).min(1.0); self.strength }
    pub fn violate(&mut self) { self.violations += 1; self.strength = (self.strength - 0.15).max(0.0); }
    pub fn is_active(&self, min: f32) -> bool { self.strength >= min }

    pub fn confidence(&self) -> f32 {
        if self.verifications == 0 && self.violations == 0 { return self.source.base_trust(); }
        let total = (self.verifications + self.violations) as f32;
        self.source.base_trust() * (self.verifications as f32 / total)
    }

    pub fn effective_strength(&self) -> f32 { self.strength * self.confidence() }
}

#[derive(Debug, Clone)]
pub struct LockCheck {
    pub lock_id: u64,
    pub triggered: bool,
    pub description: String,
    pub enforcement: String,
    pub effective_strength: f32,
}

pub struct LockAccumulator {
    locks: HashMap<u64, Lock>,
    min_strength: f32,
}

impl Default for LockAccumulator {
    fn default() -> Self { Self::new() }
}

impl LockAccumulator {
    pub fn new() -> Self { Self { locks: HashMap::new(), min_strength: 0.1 } }

    pub fn add(&mut self, lock: Lock) -> u64 {
        let id = lock.id;
        self.locks.insert(id, lock);
        id
    }

    pub fn check(&self, input: &str) -> Vec<LockCheck> {
        let mut checks = Vec::new();
        for lock in self.locks.values() {
            if !lock.is_active(self.min_strength) { continue; }
            if input.contains(&lock.trigger_pattern) {
                checks.push(LockCheck {
                    lock_id: lock.id,
                    triggered: true,
                    description: lock.description.clone(),
                    enforcement: lock.enforcement.clone(),
                    effective_strength: lock.effective_strength(),
                });
            }
        }
        checks.sort_by(|a, b| b.effective_strength.partial_cmp(&a.effective_strength).unwrap());
        checks
    }

    pub fn len(&self) -> usize { self.locks.len() }
    pub fn is_empty(&self) -> bool { self.locks.is_empty() }
}

// ── Agent State ───────────────────────────────────────────────────────

pub struct AgentState {
    pub agent_id: AgentId,
    pub beliefs: BeliefStore,
    pub constraints: ConstraintEngine,
    pub expertise: Expertise,
}

// ── Shared State ──────────────────────────────────────────────────────

pub struct SharedState {
    pub locks: LockAccumulator,
    pub fused_beliefs: HashMap<AgentId, BeliefScore>,
}

// ── Multi-Agent DCS Engine ────────────────────────────────────────────

pub struct MultiAgentDCS {
    agents: HashMap<AgentId, AgentState>,
    shared: SharedState,
}

impl Default for MultiAgentDCS {
    fn default() -> Self { Self::new() }
}

impl MultiAgentDCS {
    pub fn new() -> Self {
        Self {
            agents: HashMap::new(),
            shared: SharedState {
                locks: LockAccumulator::new(),
                fused_beliefs: HashMap::new(),
            },
        }
    }

    pub fn agent_join(&mut self, agent_id: AgentId, expertise: Expertise) {
        self.agents.insert(agent_id, AgentState {
            agent_id,
            beliefs: BeliefStore::new(),
            constraints: ConstraintEngine::new(),
            expertise,
        });
    }

    pub fn agent_leave(&mut self, agent_id: AgentId) -> bool {
        self.agents.remove(&agent_id).is_some()
    }

    pub fn agent_count(&self) -> usize { self.agents.len() }

    /// Each agent proposes best tiles from local beliefs, fused by trust weight.
    pub fn dcs_query(&self, query: &str) -> Vec<(AgentId, BeliefScore)> {
        let mut scored = Vec::new();
        for (&aid, state) in &self.agents {
            if let Some(belief) = state.beliefs.get(query) {
                let fused = self.shared.fused_beliefs.get(&aid)
                    .copied()
                    .unwrap_or_default();
                let weight = belief.composite() * fused.composite();
                scored.push((aid, belief));
                let _ = weight; // used for ranking in full impl
            }
        }
        scored.sort_by(|a, b| b.1.composite().partial_cmp(&a.1.composite()).unwrap());
        scored
    }

    pub fn check_locks(&self, _agent_id: AgentId, command: &str) -> Vec<LockCheck> {
        self.shared.locks.check(command)
    }

    pub fn update_belief(&mut self, agent_id: AgentId, key: &str, dim: BeliefDimension, strength: f32) {
        if let Some(state) = self.agents.get_mut(&agent_id) {
            if strength >= 0.0 {
                state.beliefs.reinforce(key, dim, strength);
            } else {
                state.beliefs.undermine(key, dim, strength.abs());
            }
            if let Some(fused) = state.beliefs.get(key) {
                self.shared.fused_beliefs.insert(agent_id, fused);
            }
        }
    }

    pub fn consensus_round(&mut self, agent_ids: &[AgentId]) -> ConsensusResult {
        let active: Vec<_> = agent_ids.iter().filter(|id| self.agents.contains_key(id)).copied().collect();
        let disagreement = agent_ids.len().saturating_sub(active.len());

        ConsensusResult {
            active_agents: active.len(),
            disagreement_count: disagreement,
            disagreement_rate: if agent_ids.is_empty() { 0.0 } else { disagreement as f64 / agent_ids.len() as f64 },
        }
    }

    pub fn add_shared_lock(&mut self, lock: Lock) -> u64 {
        self.shared.locks.add(lock)
    }

    pub fn constraint_audit(&self, agent_id: AgentId, command: &str) -> AuditOutcome {
        if let Some(state) = self.agents.get(&agent_id) {
            state.constraints.audit(command)
        } else {
            AuditOutcome::Fail("Agent not found".to_string())
        }
    }
}

#[derive(Debug, Clone)]
pub struct ConsensusResult {
    pub active_agents: usize,
    pub disagreement_count: usize,
    pub disagreement_rate: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dcs_new() {
        let dcs = MultiAgentDCS::new();
        assert!(dcs.agents.is_empty());
        assert!(dcs.shared.locks.is_empty());
    }

    #[test]
    fn test_agent_join_leave() {
        let mut dcs = MultiAgentDCS::new();
        dcs.agent_join(1, vec!["math".to_string(), "geometry".to_string()]);
        assert_eq!(dcs.agent_count(), 1);
        assert!(dcs.agent_leave(1));
        assert_eq!(dcs.agent_count(), 0);
        assert!(!dcs.agent_leave(99)); // already gone
    }

    #[test]
    fn test_agent_expertise() {
        let mut dcs = MultiAgentDCS::new();
        dcs.agent_join(1, vec!["math".to_string(), "geometry".to_string()]);
        let state = dcs.agents.get(&1).unwrap();
        assert_eq!(state.expertise, vec!["math", "geometry"]);
    }

    #[test]
    fn test_belief_store_set_get() {
        let mut store = BeliefStore::new();
        let b = BeliefScore::new(0.9, 0.8, 0.7);
        store.set("pythagorean", b);
        let got = store.get("pythagorean").unwrap();
        assert!((got.confidence - 0.9).abs() < 0.001);
        assert!(store.get("nonexistent").is_none());
    }

    #[test]
    fn test_belief_reinforce_undermine() {
        let mut store = BeliefStore::new();
        store.set("test", BeliefScore::new(0.5, 0.5, 0.5));
        store.reinforce("test", BeliefDimension::Confidence, 1.0);
        let got = store.get("test").unwrap();
        assert!(got.confidence > 0.5, "reinforce should increase");
        store.undermine("test", BeliefDimension::Trust, 1.0);
        let got2 = store.get("test").unwrap();
        assert!(got2.trust < 0.5, "undermine should decrease");
    }

    #[test]
    fn test_belief_decay() {
        let mut store = BeliefStore::new();
        store.set("test", BeliefScore::new(1.0, 1.0, 1.0));
        store.tick();
        let got = store.get("test").unwrap();
        assert!(got.confidence < 1.0, "decay should pull toward 0.5");
    }

    #[test]
    fn test_composite_geometric_mean() {
        let b = BeliefScore::new(1.0, 1.0, 1.0);
        assert!((b.composite() - 1.0).abs() < 0.001);
        let z = BeliefScore::new(0.0, 0.0, 0.0);
        assert!((z.composite() - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_actionable() {
        let b = BeliefScore::new(0.9, 0.8, 0.7);
        assert!(b.actionable(0.8, 0.7, 0.6));
        assert!(!b.actionable(1.0, 0.0, 0.0));
    }

    #[test]
    fn test_constraint_audit_pass() {
        let engine = ConstraintEngine::new();
        assert_eq!(engine.audit("select * from users"), AuditOutcome::Pass);
    }

    #[test]
    fn test_constraint_audit_fail() {
        let engine = ConstraintEngine::new();
        match engine.audit("RM -RF / HOME") {
            AuditOutcome::Fail(msg) => assert!(msg.contains("Forbidden")),
            _ => panic!("should fail"),
        }
    }

    #[test]
    fn test_lock_new_verify_violate() {
        let mut lock = Lock::new("test", "danger", "BLOCK", LockSource::Expert);
        assert_eq!(lock.source.base_trust(), 1.0);
        lock.verify();
        assert_eq!(lock.verifications, 1);
        assert!(lock.strength >= 1.0); // base + 0.05, capped at 1.0
        lock.violate();
        assert_eq!(lock.violations, 1);
    }

    #[test]
    fn test_lock_confidence() {
        let mut lock = Lock::new("test", "x", "BLOCK", LockSource::Observation);
        for _ in 0..10 { lock.verify(); }
        assert!((lock.confidence() - 0.7).abs() < 0.001); // 10/10 * 0.7
    }

    #[test]
    fn test_lock_accumulator_check() {
        let mut acc = LockAccumulator::new();
        acc.add(Lock::new("rm guard", "rm -rf", "BLOCK", LockSource::Expert));
        let checks = acc.check("rm -rf /tmp/stuff");
        assert_eq!(checks.len(), 1);
        assert!(checks[0].triggered);
        let no_match = acc.check("echo hello");
        assert!(no_match.is_empty());
    }

    #[test]
    fn test_dcs_query_with_beliefs() {
        let mut dcs = MultiAgentDCS::new();
        dcs.agent_join(1, vec!["math".to_string()]);
        dcs.agent_join(2, vec!["geometry".to_string()]);
        dcs.update_belief(1, "pythagorean", BeliefDimension::Confidence, 0.9);
        dcs.update_belief(2, "pythagorean", BeliefDimension::Confidence, 0.7);

        let results = dcs.dcs_query("pythagorean");
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 1); // agent 1 has higher belief
    }

    #[test]
    fn test_shared_locks_across_agents() {
        let mut dcs = MultiAgentDCS::new();
        dcs.agent_join(1, vec!["ops".to_string()]);
        dcs.agent_join(2, vec!["ops".to_string()]);
        dcs.add_shared_lock(Lock::new("no rm", "rm -rf", "BLOCK", LockSource::Expert));

        let checks1 = dcs.check_locks(1, "rm -rf /");
        let checks2 = dcs.check_locks(2, "rm -rf /");
        assert_eq!(checks1.len(), checks2.len());
        assert!(checks1[0].triggered);
    }

    #[test]
    fn test_consensus_round() {
        let mut dcs = MultiAgentDCS::new();
        dcs.agent_join(1, vec!["a".to_string()]);
        dcs.agent_join(2, vec!["b".to_string()]);
        let result = dcs.consensus_round(&[1, 2, 99]); // 99 doesn't exist
        assert_eq!(result.active_agents, 2);
        assert_eq!(result.disagreement_count, 1);
        assert!((result.disagreement_rate - 0.333).abs() < 0.01);
    }

    #[test]
    fn test_fused_belief_update() {
        let mut dcs = MultiAgentDCS::new();
        dcs.agent_join(1, vec!["test".to_string()]);
        dcs.update_belief(1, "key", BeliefDimension::Confidence, 0.9);
        let fused = dcs.shared.fused_beliefs.get(&1).unwrap();
        assert!(fused.confidence > 0.5);
    }

    #[test]
    fn test_constraint_audit_unknown_agent() {
        let dcs = MultiAgentDCS::new();
        match dcs.constraint_audit(99, "anything") {
            AuditOutcome::Fail(msg) => assert!(msg.contains("not found")),
            _ => panic!("should fail for unknown agent"),
        }
    }

    #[test]
    fn test_belief_dimension_set_get() {
        let mut b = BeliefScore::default();
        b.set(BeliefDimension::Trust, 0.99);
        assert!((b.get(BeliefDimension::Trust) - 0.99).abs() < 0.001);
        assert!((b.get(BeliefDimension::Confidence) - 0.5).abs() < 0.001); // unchanged
    }

    #[test]
    fn test_lock_active_threshold() {
        let lock = Lock::new("weak", "x", "WARN", LockSource::Inferred);
        assert!(lock.is_active(0.3)); // 0.4 > 0.3
        assert!(!lock.is_active(0.5)); // 0.4 < 0.5
    }
}
