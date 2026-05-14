import asyncio


def test_agent_startup_safety_and_lazy_runtime():
    from core.agent.network_agent import NetworkAgent
    from core.agent.notification_agent import NotificationAgent
    from core.agent.scheduler_agent import SchedulerAgent
    from core.agent.nlp_agent import NLPAgent

    # These constructors must be safe without an already-running event loop.
    net = NetworkAgent({"queue_enabled": True})
    notif = NotificationAgent({})
    sched = SchedulerAgent({"persist_enabled": False})

    assert net is not None
    assert notif is not None
    assert sched is not None

    # Path import regression guard: should not raise NameError in _init_gguf_model.
    nlp = NLPAgent({"use_gguf": False, "use_transformers": False, "use_spacy": False})
    assert nlp is not None

    async def _runtime_checks():
        # Network session can be created lazily inside an async context.
        session = await net.get_session("default")
        assert session is not None
        await net.close()

        # Notification queue starts/stops safely once loop is running.
        notif.start_queue_processor()
        await notif.stop_queue_processor()

        # Scheduler can start lazily and accept a schedule.
        result = await sched.add_delayed_schedule(
            name="startup-safety-smoke",
            delay_seconds=60,
            task_data={"test": True},
        )
        assert result.get("success") is True
        await sched.shutdown()

    asyncio.run(_runtime_checks())
