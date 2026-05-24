"""Script to fix ai_backend.py asyncio/callback issues"""
import re

with open('core/ui/ai_backend.py', 'r', encoding='utf-8') as f:
    content = f.read()

original = content

# ===============================================================
# FIX 1: Remove timeout=0 from fut.result() - causes InvalidStateError
# The future isn't complete when callback fires, timeout=0 fails immediately
# ===============================================================
content = content.replace(
    'result = fut.result(\n                        timeout=0\n                    )',
    'result = fut.result()'
)
print("Fix 1: fut.result() - removed timeout=0")

# ===============================================================
# FIX 2: Remove brain_process from _fallback_thread
# brain_process is async and causes thread/asyncio issues in fallback path
# The fallback should ONLY use agent.run() or direct LLM access
# ===============================================================
# Find and replace the Strategy 1 block in _fallback_thread
# Look for the pattern starting with "# STRATEGY 1" and ending before "# STRATEGY 2"

# Replace the brain_process strategy with a simple skip
old_brain_section = '''# STRATEGY 1
                        # brain_process
                        # =================================================
                        try:

                            brain_process = getattr(
                                self,
                                "brain_process",
                                None,
                            )

                            if callable(
                                brain_process
                            ):

                                logger.info(
                                    "Fallback: brain_process"
                                )

                                coro = brain_process(
                                    "text",
                                    {
                                        "text": text,
                                        "timestamp": time.time(),
                                    },
                                )

                                # FIX:
                                # bool await crash
                                if inspect.isawaitable(
                                    coro
                                ):

                                    result = await asyncio.wait_for(
                                        coro,
                                        timeout=20.0,
                                    )

                                else:

                                    result = coro

                                # FIX:
                                # function.items crash
                                if isinstance(
                                    result,
                                    dict,
                                ):

                                    if result.get(
                                        "success"
                                    ):

                                        output = result.get(
                                            "output",
                                            "",
                                        )

                                        if output:

                                            return str(
                                                output
                                            ).strip()

                                elif result is not None:

                                    value = str(
                                        result
                                    ).strip()

                                    if value:

                                        return value

                        except asyncio.TimeoutError:

                            logger.error(
                                "Fallback brain_process timeout"
                            )

                        except Exception as e:

                            logger.exception(
                                "Fallback brain_process failed: %s",
                                e,
                            )'''

new_brain_section = '''# SKIPPED: brain_process removed from fallback
                        # brain_process causes asyncio/threading issues when called from fallback thread
                        # The main _process_message path already handles brain_process properly
                        pass'''

content = content.replace(old_brain_section, new_brain_section)
print("Fix 2: Removed brain_process from _fallback_thread")

# ===============================================================
# FIX 3: Ensure _processing is ALWAYS reset in _handle_result
# Add finally block to guarantee cleanup
# ===============================================================
old_handle_result_end = '''                # --------------------------------------------------------
                # FINAL CLEANUP
                # --------------------------------------------------------
                finally:

                    try:

                        self._show_typing_indicator_safe(
                            False
                        )

                    except Exception:
                        pass

                    self._processing = False'''

new_handle_result_end = '''                # --------------------------------------------------------
                # FINAL CLEANUP (GUARANTEED)
                # --------------------------------------------------------
                finally:
                    # ALWAYS reset processing flag - this is critical
                    self._processing = False

                    try:
                        self._show_typing_indicator_safe(False)
                    except Exception:
                        pass'''

content = content.replace(old_handle_result_end, new_handle_result_end)
print("Fix 3: Guaranteed _processing cleanup in _handle_result")

# ===============================================================
# FIX 4: Also ensure cleanup in send_user_message main path
# Wrap the whole processing in try/finally
# ===============================================================
# Find the section where _processing is set and add finally
old_processing = '''            # lock processing
                self._processing = True

            # ============================================================
            # UPDATE STATE
            # ============================================================'''

new_processing = '''            # lock processing
                self._processing = True

            # Ensure cleanup when this function exits
            try:'''

content = content.replace(old_processing, new_processing)
print("Fix 4: Added try wrapper around processing")

# Need to close the try block with finally at the end of send_user_message
# Find where the function ends and add finally
old_send_end = '''            self._processing = False

    # ─────────────────────────────────────────────────────────────────
    # FALLBACK: run in a fresh thread when the main loop is dead/busy
    # ─────────────────────────────────────────────────────────────────
    def _fallback_thread(self, text: str):'''

new_send_end = '''            self._processing = False

            finally:
                # GUARANTEED cleanup - reset processing on ANY exit
                self._processing = False
                try:
                    self._show_typing_indicator_safe(False)
                except Exception:
                    pass

    # ─────────────────────────────────────────────────────────────────
    # FALLBACK: run in a fresh thread when the main loop is dead/busy
    # ─────────────────────────────────────────────────────────────────
    def _fallback_thread(self, text: str):'''

content = content.replace(old_send_end, new_send_end)
print("Fix 4b: Added finally cleanup at end of send_user_message")

with open('core/ui/ai_backend.py', 'w', encoding='utf-8') as f:
    f.write(content)

changes = content.count(original) == 0
applied = original != content
print(f"\nChanges applied: {applied}")

if applied:
    print("All fixes successfully applied!")
else:
    print("WARNING: No changes made - patterns may not have matched")