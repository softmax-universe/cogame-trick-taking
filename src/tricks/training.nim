## Private authoritative decisions, separate from spectator artifacts.
import std/[json, options]
import bitworld/decision_trajectory
import llm, prompts, sim

proc recordAppliedDecision*(trajectory: DecisionTrajectory, before, after: Sim,
    seat: int, proposed: Decision, operatorPrompt: string,
    deliberateTeacher: bool, selection: PromptSelection, fallbackReason = "") =
  var applied = newJNull()
  for index in before.events.len ..< after.events.len:
    let event = after.events[index]
    if event.slot == seat and event.kind in {evBid, evPlay, evDiscard, evPass}:
      var actual = Decision(notes: event.text)
      case event.kind
      of evPlay: actual.move = playMove(event.card)
      of evDiscard: actual.move = discardMove(event.card)
      of evPass: actual.move = passMove(event.cards)
      of evBid: actual.move = bidMove(event.action, event.value, event.suit)
      else: discard
      applied = before.decisionAction(actual)
      break
  doAssert applied.kind == JObject, "engine emitted no applied decision event"
  var attempts = proposed.nativeAttempts
  var selected = none(string)
  let fallback = proposed.fallback or fallbackReason.len > 0
  if not fallback and attempts.len == 0:
    let origin = if deliberateTeacher: aoTeacher else: aoUnknown
    var evidence = newDecisionAttempt("event-" & $before.events.len & "-applied",
      if deliberateTeacher: proposed.policy else: "external-trick-taking", origin)
    if deliberateTeacher:
      evidence.prompt = %*[{"role": "system", "content": systemPrompt(before, seat, selection)},
        {"role": "user", "content": userPrompt(before, operatorPrompt)}]
      evidence.response = %($before.decisionAction(proposed))
    else:
      evidence.response = %proposed.submittedResponse
    evidence.parsedAction = before.decisionAction(proposed)
    evidence.accepted = true
    attempts.add(evidence)
  for index in 0 ..< attempts.len:
    if attempts[index].accepted:
      if fallback:
        attempts[index].accepted = false
        attempts[index].rejectionReason = some(if fallbackReason.len > 0: fallbackReason else: "consumed fallback")
      else:
        selected = some(attempts[index].attemptId)
  var observation = before.frameStateJson()
  observation["prompt_template"] = selection.promptMetadata(
    systemPrompt(before, seat, selection), userPrompt(before, operatorPrompt))
  observation.delete("tell")
  observation.delete("kitty")
  observation.delete("discard")
  for player in observation["seats"]:
    if player["slot"].getInt() != seat:
      player["hand"] = newJArray()
      player["notes"] = %""
  trajectory.recordDecision("event-" & $before.events.len, $seat,
    observation, attempts, selected, applied,
    if fallback: asFallback else: asAccepted, terminal = after.done,
    fallbackOrigin = if fallback: some(if fallbackReason.len > 0: fallbackReason else: proposed.policy) else: none(string))

proc recordFailedDecision*(trajectory: DecisionTrajectory, before: Sim,
    seat: int, failed: Decision, operatorPrompt: string,
    selection: PromptSelection) =
  var observation = before.frameStateJson()
  observation["prompt_template"] = selection.promptMetadata(
    systemPrompt(before, seat, selection), userPrompt(before, operatorPrompt))
  observation.delete("tell")
  observation.delete("kitty")
  observation.delete("discard")
  for player in observation["seats"]:
    if player["slot"].getInt() != seat:
      player["hand"] = newJArray()
      player["notes"] = %""
  let status = if failed.failureKind == "player_error": asRejected else: asMissing
  trajectory.recordDecision("event-" & $before.events.len, $seat,
    observation, failed.nativeAttempts, none(string), newJNull(),
    status, terminal = true)

proc finishTrajectory*(trajectory: DecisionTrajectory, sim: Sim,
    failed = false, failureInfo: JsonNode = nil) =
  var outcome =
    if failed:
      %*{
        "failed": true,
        "module": sim.module,
        "handsPlayed": sim.handsPlayed,
        "handsScored": sim.handsScored,
        "seed": sim.config.seed,
        "failure": (if failureInfo.isNil: newJNull() else: failureInfo)
      }
    else:
      sim.resultsJson()
  var participants = newJObject()
  if not failed:
    for seat in 0 ..< sim.config.players.len:
      participants[$seat] = %*{"score": outcome["scores"][seat]}
  let status =
    if failed: esFailed
    elif sim.reason == "complete": esCompleted
    else: esTruncated
  trajectory.finish(status, outcome, participants)
