package toki.falldetector;

import net.fabricmc.api.ModInitializer;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import net.fabricmc.fabric.api.client.keybinding.v1.KeyBindingHelper;
import net.minecraft.client.util.InputUtil;

import org.lwjgl.glfw.GLFW;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class Falldetector implements ModInitializer {
	public static final String MOD_ID = "falldetector";

	// This logger is used to write text to the console and the log file.
	// It is considered best practice to use your mod id as the logger's name.
	// That way, it's clear which mod wrote info, warnings, and errors.
	public static final Logger LOGGER = LoggerFactory.getLogger(MOD_ID);

	@Override
	public void onInitialize() {
		// This code runs as soon as Minecraft is in a mod-load-ready state.
		// However, some things (like resources) may still be uninitialized.
		// Proceed with mild caution.

		// Get player position data each in game tick
		ClientTickEvents.END_CLIENT_TICK.register(client -> {
    		if (client.player != null) {
        		var player = client.player;
        		var posY = player.getY();
        		var velocity = player.getVelocity();
        		var onGround = player.isOnGround();
        		System.out.println("Y: " + posY + " | Vel: " + velocity + " | onGround: " + onGround);
    		}
		});


		LOGGER.info("Hello Fabric world!");
	}

	

}