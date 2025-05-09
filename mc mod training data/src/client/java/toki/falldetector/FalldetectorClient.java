package toki.falldetector;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

import org.lwjgl.glfw.GLFW;

import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import net.fabricmc.fabric.api.client.keybinding.v1.KeyBindingHelper;
import net.fabricmc.loader.api.FabricLoader;
import net.minecraft.client.network.ClientPlayerEntity;
import net.minecraft.client.option.KeyBinding;
import net.minecraft.client.util.InputUtil;
import net.minecraft.text.Text;
import net.minecraft.util.math.Vec3d;

public class FalldetectorClient implements ClientModInitializer {

    private static KeyBinding keyBinding;
	private int falls;
    private int tick = 0;

    @Override
    public void onInitializeClient() {
        
        // Create new keybinding
        keyBinding = KeyBindingHelper.registerKeyBinding(new KeyBinding(
            "key.falldetector.fall", // The translation key of the keybinding's name
            InputUtil.Type.KEYSYM, // The type of the keybinding, KEYSYM for keyboard, MOUSE for mouse.
            GLFW.GLFW_KEY_R, // The keycode of the key
            "category.falldetector.test" // The translation key of the keybinding's category.
        ));

		Path file = newSession();

        // Get player position data each in game tick
        ClientTickEvents.END_CLIENT_TICK.register(client -> {
    		if (client.player != null) {
                ++tick;
                ClientPlayerEntity player = client.player;

                boolean isFall = false;
                if (keyBinding.wasPressed()) {
                    isFall = true;
					++falls;
                    client.player.sendMessage(Text.literal("---------- Fall: [ " + falls + " ]----------"), false);
                }
        		
        		double posY = player.getY();
				double posX = player.getX();
				double posZ = player.getZ();
        		Vec3d vel = player.getVelocity();
        		boolean onGround = player.isOnGround();

                String line = tick + "," + posX + "," + posY + "," + posZ + "," + vel.x + "," + vel.y + "," + vel.z + "," + onGround + "," + isFall + "\n";
                if (!writeDataToFile(line, file)) {
					client.player.sendMessage(Text.literal("Error: could not write to file!"), false);
				}
            }
		});

    }

	// Create new session csv file
	private Path newSession() {
		try {
			Path dir = FabricLoader.getInstance().getGameDir().resolve("data/");
            Files.createDirectories(dir);  // makes sure the folder exists

			DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd_HH-mm-ss");
			String timestamp = LocalDateTime.now().format(formatter);
            Path file = dir.resolve("fall_data_" + timestamp + ".csv");

			// Write header if file doesn't exist
            if (!Files.exists(file)) {
                Files.writeString(file, "tick,x,y,z,velX,velY,velZ,onGround,isFall\n", StandardOpenOption.CREATE);
            }

			return file;
		}
		catch (IOException e) {
			e.printStackTrace();
		}

		// Error
		return null;
	}

    // Writes player position data to a csv file
    private boolean writeDataToFile(String line, Path file) {
		if (file != null) {
			try {
            	// Append new line to csv file
            	Files.writeString(file, line, StandardOpenOption.CREATE, StandardOpenOption.APPEND);
				return true;
        	} 
			catch (IOException e) {
            	e.printStackTrace();
        	}
		}
		return false;
    }
    
}
