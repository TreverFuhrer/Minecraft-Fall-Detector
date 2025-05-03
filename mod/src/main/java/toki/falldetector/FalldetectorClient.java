package toki.falldetector;

import org.lwjgl.glfw.GLFW;

import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import net.fabricmc.fabric.api.client.keybinding.v1.KeyBindingHelper;
import net.minecraft.client.option.KeyBinding;
import net.minecraft.client.util.InputUtil;
import net.minecraft.text.Text;

public class FalldetectorClient implements ClientModInitializer {

    private static KeyBinding keyBinding;

    @Override
    public void onInitializeClient() {
        
        // Create new keybinding
        keyBinding = KeyBindingHelper.registerKeyBinding(new KeyBinding(
            "key.falldetector.fall", // The translation key of the keybinding's name
            InputUtil.Type.KEYSYM, // The type of the keybinding, KEYSYM for keyboard, MOUSE for mouse.
            GLFW.GLFW_KEY_R, // The keycode of the key
            "category.falldetector.test" // The translation key of the keybinding's category.
        ));

        // Do when key binding is pressed
        ClientTickEvents.END_CLIENT_TICK.register(client -> {
            while (keyBinding.wasPressed()) {
	        client.player.sendMessage(Text.literal("Fall key was pressed!"), false);
            }
        });

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
        
    }
    
}
